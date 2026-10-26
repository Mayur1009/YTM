Backends
========

Core
----

.. currentmodule:: ytm._core.backends

.. autosummary::
   :toctree: generated

   base.BaseDevice
   cpu.CPUDevice
   cuda.CUDADevice
   cuda.Kernel
   cuda.Launcher
   buffers.FitBuffers
   buffers.PackedClauses
   toolchain.Toolchain
   toolchain.compiler_cmd

Tsetlin Machine
---------------

.. currentmodule:: ytm.tm.backends

.. autosummary::
   :toctree: generated

   cpu.CPUDevice
   cuda.CUDADevice
   buffers.TMFitBuffers

Loss-Guided TM
--------------

.. currentmodule:: ytm.guided.backends

.. autosummary::
   :toctree: generated

   cpu.CPUDevice
   cuda.CUDADevice
   buffers.GuidedFitBuffers
