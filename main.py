from framework import Framework
import os

def load_models(directory):
        framework = Framework()

        filenames = os.listdir(directory)
        for filename in filenames: 
            if filename.endswith(".py") and filename != "__init__.py":
                model_file = filename[:-3]
                model = __import__(f"{directory}.{model_file}", fromlist=[model_file])
                model_class = getattr(model, model_file)
                model_instance = model_class()
                framework.add_model(model_instance)
        
        return framework

def print_banner():
    banner = """
    ______________
    < scikitsploit >
    --------------
            \   ^__^
            \  (oo)\_______
                (__)\       )\/\\
                    ||----w |
                    ||     ||
    """

    print(banner)

if __name__ == "__main__":
    models_directory = "models"
    my_framework = load_models(models_directory)
    print_banner()

    while True:
        my_framework.show_prompt()
        command = input()
        exit_requested = my_framework.handle_command(command)
        if exit_requested:
            break
