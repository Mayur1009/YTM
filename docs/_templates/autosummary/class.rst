{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{%- if objname.endswith("Config") %}
   :member-order: bysource
   :no-inherited-members:
{%- endif %}
