# pulse-benchmarks
Some basic benchmarks for `pennylane.pulse` for public display.

We run the basic benchmark of evolving a state according to

$$ H(t) = \sum_q \omega_q Z_q + \sum_q g_q Y_q \otimes Y_{q+1} +  \sum_q f_q(t) X_q \otimes X_{q+1} $$

and then computing the expectation value of 

$$ H_\text{obj} = \sum_q c^X_q X_q \otimes X_{q+1} + c^Y_q Y_q \otimes Y_{q+1}. $$

As time-dependent envelopes $f_q(t)$, we use a (modulated) Gaussian in resonance with the qubit frequencies $\omega_q = 5$ GHz.

![Benchmark results comparing PennyLane, qiskit-dynamics, and QuTiP.](/plots/Forward-pass_PL_QUTIP_QISKIT.png)

The main benchmark is done in [Forward-pass_PL_QUTIP_QISKIT_M1.ipynb](Forward-pass_PL_QUTIP_QISKIT_M1.ipynb). We ran this on an Macbook Pro 16 with an M1 pro chip. We get qualitatively identical results using an Intel i7-1260P in an LG gram 16 (2022), see [Forward-pass_PL_QUTIP_QISKIT_i7-1260P.ipynb](Forward-pass_PL_QUTIP_QISKIT_i7-1260P.ipynb).  

We also perform the same benchmark on a NVIDIA TITAN Xp GPU, see [Forward-pass_PL_hardware.ipynb](Forward-pass_PL_hardware.ipynb).


## Disclaimer

We compare to third party repositories offering similar functionality. Benchmarks never capture the full capability of either repository and are always specific to the choice of benchmark problem. Further, we cannot guarantee using the latest and best standards for any third party repository that we compare to. We invite experts to submit suggestions for better performance by opening Issues and/or Pull Requests.
