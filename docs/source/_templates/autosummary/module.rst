{%- set small_name = fullname.split('.')[-1] %}

{{ small_name | escape | underline }}

This is summary documentation for {{ fullname | escape }} module.
:doc:`Click here for full documentation. </api_documentation/{{fullname}}>`


{% if modules %}
.. toctree::
   :caption: API Summary for sub-modules:
   :maxdepth: 1


{% for item in modules %}

   {%- set item_small_name = item.split('.')[-1] %}
   {%- set item_summary_name = 'summary.' + item %}
   {{ fullname + '.' + item}}


{%- endfor %}
{% endif %}


.. rubric:: Links to Full Documentation:
   :heading-level: 1

.. py:currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
   :noindex:

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:

{% for item in modules %}

   {{item}}

{%- endfor %}
{% endif %}


{%- endblock %}

