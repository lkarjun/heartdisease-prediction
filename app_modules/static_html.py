import streamlit.components.v1 as components
from jinja2 import Environment, BaseLoader

template = Environment(loader=BaseLoader)


track_model_html = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-0evHe/X+R7YkIZDRvuzKMRqM+OrBnVFBL6DOitfPri4tjfHxaWutUpFmBp4vmVor" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/js/bootstrap.bundle.min.js" integrity="sha384-pprn3073KE6tl6bjs2QrFaJGz5/SUsLqktiwsUTF55Jfv3qYSDhgCecCxMW52nD2" crossorigin="anonymous"></script>
<div style='font-size: 18px;'>

    {% if notdetails %}

    <div class='text-center'>
        <div class='m-3'>
            <span class="badge" style='font-size: 20px; background-color: #DB6B97'>{{name}} v{{version}}</span>
        </div>
    
    {% else %}
        <div class='text-left'>
    {% endif %}

        <div class='m-2'>
            {% if latest %}
                <span class="badge rounded-pill" style="background-color: #5FD068; color: #121212">Latest</span>
            {% else %}
                <span class="badge rounded-pill text-bg-danger">Outdated</span>
            {% endif %}
            <span class="badge rounded-pill text-bg-info">{{ data['version'] }}</span>
            <span class="badge rounded-pill text-bg-info">{{ data['updated_time'] }}</span>
            {% for _, v in data['tag'].items() %}
                <span class="badge rounded-pill text-bg-warning">{{ v }}</span>
            {% endfor %}
        </div>
        
        <div class='m-2'>
            <span class="badge" style="background-color: #ECDBBA; color: #121212">{{ data['run_id'] }}</span>
            {% if latest %}
                <span class="badge" style="background-color: #5FD068; color: #121212">{{ stage }}</span>
                <span class="badge" style="background-color: #5FD068; color: #121212">{{ status }}</span>
            {% endif %}
        </div>
    </div>
    <br>

</div>

"""

home_version_display = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-0evHe/X+R7YkIZDRvuzKMRqM+OrBnVFBL6DOitfPri4tjfHxaWutUpFmBp4vmVor" crossorigin="anonymous">
<div style='font-size: 18px;'>
    <div class='text-center'>
        <span class="badge text-bg-info">{{ tag }} v{{ v }}.0</span>
    </div>
</div>
"""



def render(html, height = 60, **kwargs):
    rendered_file = template.from_string(html).render(**kwargs)
    components.html(rendered_file, height=height)
