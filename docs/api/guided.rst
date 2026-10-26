Loss-Guided TM
==============

Models
------

.. currentmodule:: ytm.guided

.. autosummary::
   :toctree: generated

   MultiClassTM
   MultiOutputTM
   BinaryTM
   RegressionTM

Configuration
-------------

.. currentmodule:: ytm.guided.config

.. autosummary::
   :toctree: generated

   TMConfig
   ActFn
   FbSignal

Activations and losses
----------------------

.. currentmodule:: ytm.guided.backends.act_loss

.. autosummary::
   :toctree: generated

   ActLoss
   SoftmaxCE
   SigmoidBCE
   MSE
   MAE
   Huber
   ASL
   SCE
   Tversky
   FocalBCE
   FocalCE
