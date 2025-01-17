import jinja2

env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))

def get_agent_prompt() -> str:
    template = env.get_template("prompt.jinja2")
    agent_prompt = template.render()

    return agent_prompt
