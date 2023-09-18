import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

from time import time
from datetime import datetime

import matplotlib.pyplot as plt

import qutip as qp
from qiskit_dynamics.array import Array
Array.set_default_backend('jax')
from qiskit.quantum_info import Operator
from qiskit_dynamics import Solver
from qiskit_dynamics import DiscreteSignal

omega0 = 5.
g0 = 0.01
t_span = jnp.linspace(0., 20., 500)
atol=1e-8

# Control
def amp(p, t):
    sigma = 2.
    # return qml.pulse.pwc((t_span[0], t_span[-1]))(p, t) * jnp.sin(omega[0]*t)
    return p[0] * jnp.exp(-0.5 * ((t - 10.)/sigma)**2) /sigma/jnp.sqrt(2*jnp.pi) * jnp.sin(omega0*t)


def create_evolve(n_wires, mode="PennyLane"):
    omega = omega0 * jnp.ones(n_wires, dtype=float)
    g = g0 * jnp.ones(n_wires, dtype=float)

    # Drift
    Hdrift = [qml.PauliZ(i) for i in range(n_wires)]
    Hdrift += [qml.PauliY(i) @ qml.PauliY((i+1)%n_wires) for i in range(n_wires)]
    cdrift = jnp.concatenate([omega, g])
    H_D = qml.Hamiltonian(cdrift, Hdrift)

    fs = [amp for _ in range(n_wires)]
    Hcontrol = [qml.PauliX(i) @ qml.PauliX((i+1)%n_wires) for i in range(n_wires)]
    H_C = qml.dot(fs, Hcontrol)
    H_pulse = H_D + H_C

    # H_obj
    ops2 = [qml.PauliX(i) @ qml.PauliX((i+1)%n_wires) for i in range(n_wires)]
    ops2 += [qml.PauliY(i) @ qml.PauliY((i+1)%n_wires) for i in range(n_wires)]

    key = jax.random.PRNGKey(1337)
    coeff = jax.random.uniform(key, shape=(len(ops2),))

    H_obj = qml.Hamiltonian(coeff, ops2)
    H_obj_m = jnp.array(qml.matrix(H_obj))

    if mode=="PennyLane":
        dev = qml.device("default.qubit.jax",wires=range(n_wires))
        @jax.jit
        @qml.qnode(dev, interface="jax")
        def evolve(params):
            qml.pulse.ParametrizedEvolution(H_pulse, params, [t_span[0], t_span[-1]], atol=atol)
            return qml.expval(H_obj)

    elif mode=="qutip":
        H_drift = qp.Qobj(np.array(qml.matrix(H_D)))
        def w(i):
            def wrapped(t, args):
                v = args[f"v{i}"]
                return qml.pulse.pwc((t_span[0], t_span[-1]))(v, t) * np.sin(omega[0]*t)
            return wrapped

        H_ts = [qp.Qobj(qml.matrix(op, wire_order=range(n_wires))) for op in Hcontrol]

        H_control = [[H_ts[i], w(i)] for i in range(n_wires)]
        H_pulse_q = qp.QobjEvo([H_drift] + H_control)

        psi0 = np.eye(2**n_wires, dtype=complex)[0]
        psi0 = qp.Qobj(psi0)
        def evolve(params):
            params = np.array(params)
            args = {f"v{i}": params[i] for i in range(n_wires)} 

            # tlist = [t_span[0], t_span[-1]]
            options = qp.solver.Options(nsteps=100000, atol=atol) # necessary to increase max number of steps, otherwise error message
            output = qp.sesolve(H_pulse_q, psi0, [t_span[0], t_span[-1]], args=args, options=options)
            v = qp.Qobj(output.states[-1]).full()
            res = v.conj().T @ H_obj_m @ v
            return np.real(res)
        
    elif mode=="qiskit-dynamics":
        """
        Taken from https://qiskit.org/ecosystem/dynamics/tutorials/optimizing_pulse_sequence.html
        """
        static_hamiltonian = Operator(qml.matrix(H_D))
        drive_terms = [Operator(qml.matrix(op, wire_order=range(n_wires))) for op in Hcontrol]

        ham_solver = Solver(
            hamiltonian_operators=drive_terms,
            static_hamiltonian=static_hamiltonian,
        )
        dt = np.diff(t_span)[0]
        @jax.jit
        def evolve(params):
            signals = [DiscreteSignal(dt=dt, samples=Array(amp(params[i], t_span))) for i in range(n_wires)]

            results = ham_solver.solve(
                y0=np.eye(2**n_wires, dtype=complex)[0],
                t_span=[t_span[0], t_span[-1]],
                signals=signals,
                method='jax_odeint',
                atol=atol,
            )
            v = results.y[-1]
            return jnp.real(v.conj().T @ H_obj_m @ v)

    return evolve

def _timeit(callable, *args, reps=3):

    jittime0 = time()
    res0 = callable(*args)
    dt_jit = time() - jittime0 

    dts = []
    for k in range(reps):
        t0 = time()
        resi = callable(*args)
        dt = time() - t0

        dts.append(dt)
    
    return dt_jit, np.mean(dts), np.std(dts)
def _timeit_jax(callable, *args, reps=3):
    callable.clear_cache()

    jittime0 = time()
    res0 = jax.block_until_ready(callable(*args))
    dt_jit = time() - jittime0 

    dts = []
    for k in range(reps):
        t0 = time()
        resi = jax.block_until_ready(callable(*args))
        dt = time() - t0

        dts.append(dt)
    
    return dt_jit, np.mean(dts), np.std(dts)
modes = ["PennyLane"]
reps = 3

n_wiress = list(range(2,13))

datan = {}

for mode in modes:
    dt_jit = []
    dtss = []
    ddtss = []
    for i, n_wires in enumerate(n_wiress):
        callable_ = create_evolve(n_wires, mode=mode)

        t_bins = 1
        theta = 0.5 * jnp.ones((n_wires, t_bins))

        if mode=="PennyLane" or mode=="qiskit-dynamics":
            a, b, c = _timeit_jax(callable_, theta, reps=reps)
        if mode=="qutip":
            if n_wires > 6:
                continue
            a, b, c = _timeit(callable_, theta, reps=reps)
        dt_jit.append(a)
        dtss.append(b)
        ddtss.append(c)

        print(f"mode: {mode} - {i+1} / {len(n_wiress)}, n_wires = {n_wires} - jit: {a:.4f}, exec: {b:.4f} +/- {c:.4f}")

    datan[mode] = dict(dtss=dtss, ddtss=ddtss, dt_jit=dt_jit, n_wiress=n_wiress)
def expfit(x, y):
    y = np.log(y)
    coeff = np.polyfit(x, y, 1) # y=c1 e^(c0 x) => logy = c0 x + log(c1)
    return coeff
name = f"data/{datetime.now().date()}_Forward-pass_PL-QUTIP-QISKIT-GPU"
print(name)
np.savez(name, **datan)