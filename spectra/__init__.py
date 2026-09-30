"""Spectral analysis of GDN transitions on StateBench programs.

The package has a CPU-only half and a GPU half:

- :mod:`spectra.replay` parses integer-code programs, recovers the ground-truth
  register state after every statement, and labels each token with its role.
  It needs only numpy and runs anywhere.
- :mod:`spectra.capture` and :mod:`spectra.run` load an ``olmo3_5_hybrid``
  checkpoint, record the GDN gates while prefilling a long prompt, branch the
  recurrent state to read the model's belief about every register at
  checkpoints, and optionally intervene on the gates. They need torch, fla,
  and the ``jopetty/transformers@olmo-3.5-hybrid-state-bench`` fork, and run as
  Beaker jobs (see :mod:`spectra.launch`).
- :mod:`spectra.plots` turns the job outputs into figures with the project's
  shared ``aesthetics`` conventions.
"""
