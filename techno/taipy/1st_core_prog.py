import taipy as tp 
from taipy import Core
from taipy import Config

"""
Implement the 
https://docs.taipy.io/en/release-3.0/getting_started/
"""

def build_message(name: str):
    return f"Hello {name}!"




if __name__ == "__main__":
    # define the scenario to execute a pipeline of tasks
    input_name_data_node_cfg = Config.configure_data_node(id="input_name")
    message_data_node_cfg = Config.configure_data_node(id="message")
    build_msg_task_cfg = Config.configure_task("build_msg", build_message, input_name_data_node_cfg, message_data_node_cfg)
    scenario_cfg = Config.configure_scenario("scenario", task_configs=[build_msg_task_cfg])
     #            Instantiate and run Core service
    Core().run()
    #            Manage scenarios and data nodes 
    hello_scenario = tp.create_scenario(scenario_cfg)
    hello_scenario.input_name.write("Taipy")
    hello_scenario.submit()
    print(hello_scenario.message.read())
