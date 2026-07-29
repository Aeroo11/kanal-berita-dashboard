{#
    Custom generic tests.

    These two are available in dbt_utils, but pulling in a package for them
    would add a `dbt deps` step to every CI run and a network dependency to a
    build that otherwise has none. They are a few lines each.

    A dbt test passes when it returns zero rows, so each one selects the rows
    that violate it.
#}

{% test not_empty_string(model, column_name) %}

    -- NOT NULL does not catch '' or '   ', and an empty headline is the one
    -- field the model genuinely cannot work without.
    select {{ column_name }}
    from {{ model }}
    where {{ column_name }} is not null
      and trim({{ column_name }}) = ''

{% endtest %}


{% test accepted_range(model, column_name, min_value=none, max_value=none, inclusive=true) %}

    select {{ column_name }}
    from {{ model }}
    where {{ column_name }} is not null
      and (
        false
        {% if min_value is not none %}
            or {{ column_name }} {{ '<' if inclusive else '<=' }} {{ min_value }}
        {% endif %}
        {% if max_value is not none %}
            or {{ column_name }} {{ '>' if inclusive else '>=' }} {{ max_value }}
        {% endif %}
      )

{% endtest %}


{% test expression_is_true(model, expression) %}

    {# Escape hatch for one-off invariants that do not deserve their own test. #}
    select *
    from {{ model }}
    where not ({{ expression }})

{% endtest %}


{% test unique_combination(model, columns) %}

    {# A composite uniqueness constraint. `unique` only covers one column, and
       a per-source-per-day table that gains a second row per key turns every
       rate in it into an average of averages. #}
    select {{ columns | join(', ') }}
    from {{ model }}
    group by {{ columns | join(', ') }}
    having count(*) > 1

{% endtest %}
