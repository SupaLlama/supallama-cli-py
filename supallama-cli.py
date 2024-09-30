import os, subprocess, sys

from typing_extensions import Annotated

from dotenv import load_dotenv

from fireworks.client import Fireworks

import typer


# Load environment variables``
load_dotenv()

# Create CLI app
app = typer.Typer()

# Create Fireworks client
client = Fireworks(api_key=os.getenv("FIREWORKS_AI_KEY")) 


@app.command()
def llama_chat(
    prompt_content: Annotated[
        str,
        typer.Argument(
            help="💬 Content to 🗣️  Say to the 🤖 Model",
            metavar="💬 Content for the 🤖 Model",
        )
    ] = "Tell me a joke"
    ):
    """
    🗣️ 💬 Chat with 🦙 Llama 3.1 8B Instruct
    """
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[{
            "role": "user",
            "content": prompt_content, 
        }],
    )
    print(response.choices[0].message.content)


@app.command()
def llama_code(
    prompt: Annotated[
        str,
        typer.Argument(
            help="💬 Prompt to 🗣️  Say to the 🤖 Model",
            metavar="💬 Prompt for the 🤖 Model",
        )
    ] = "Generate a functional react server component and react server action for a signup form with fields for the first name, last name, email, password, password confirmation fields and a submit button that invokes a react server action. Use typescript and the shadcn/ui component library for the react code.",
    code_only: bool = False,
    comment_code: bool = False,
    verbose: bool = False,
    ):
    """
    💬 Ask 🦙 Llama 3.1 8B Instruct to ✍️ write 📝 code
    """
    
    if code_only:
        prompt = f"{prompt} and only generate the code in the output"

    if comment_code:
        if verbose:
            prompt = f"{prompt} and annotate the code with verbose and explanative comments"
        else:
            prompt = f"{prompt} and annotate the code with concise but explanative comments"

    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[{
            "role": "user",
            "content": prompt, 
        }],
    )
    print(response.choices[0].message.content)


@app.command()
def llama_code_improve(
    input_file: Annotated[
        typer.FileText,
        typer.Argument(
            help="💬 Code you would like to have improved by the 🤖 Model",
            metavar="💬 Input Code for the 🤖 Model",
        )
    ] = sys.stdin,
    language: str = "TypeScript and possibly Functional React Components or possibly React Server Components or possibly React Server Actions",
    comment_code: bool = False,
    verbose: bool = False,
    ):
    """
    💬 Ask 🦙 Llama 3.1 8B Instruct to analyze and improve upon some 📝 code
    """
    """Read data from stdin or a file."""
    input_code = input_file.read()

    # Echo the input code back first so that the user can see the output easily in a diff
    print(input_code)

    prompt = ("Read over the following code and look for any errors or "
        "examples of poor coding practices. Then output the input code "
        f"itself with helpful, inline {language} comments added on how to " 
        f"any improve problematic lines of code. \n{input_code}")
    
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=[{
            "role": "user",
            "content": prompt, 
        }],
    )
    print(response.choices[0].message.content)


@app.command()
def fine_tune(
    display_name: Annotated[
        str,
        typer.Argument(
            help="💬 Dataset to use for Fine-tuning to the 🤖 Model",
            metavar="💬  for the 🤖 Model",
        )
    ] = "Unnamed Fine-tuning job",
    with_settings_file: str = "test_settings.yaml",
    ):
    """
    🛠️ 🦾📈 Fine-tune a 🤖 Model with 💽 data in a jsonl 💾 file 
    """
    result = subprocess.run(
        [
            'firectl',
            'create',
            'fine-tuning-job',
            '--settings-file',
            with_settings_file,
            '--display-name',
            display_name
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)    
    print(result.stderr)    


@app.command()
def fine_tune_status(
    fine_tuning_job_id: Annotated[
        str,
        typer.Argument(
            help="🛠 🦾📈 Fine-tuning Job ID to check on for status",
            metavar="🛠 🦾📈 Fine-tuning Job ID",
        )
    ],
    ):
    """
    Check on the Status of a 🛠️ 🦾📈 Fine-tuning Job 
    """
    result = subprocess.run(
        [
            'firectl',
            'get',
            'fine-tuning-job',
            fine_tuning_job_id
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)    
    print(result.stderr)    


@app.command()
def deploy_model(
    fine_tuned_model_id: Annotated[
        str,
        typer.Argument(
            help="Deploy a 🛠️ 🦾📈 Fine-tuned 🤖 Model",
            metavar="ID of the Fine-tuned 🤖 Model",
        )
    ],
    ):
    """
    Deploy a 🛠️ 🦾📈 Fine-tuned 🤖 Model to a Serverless Endpoint
    """
    result = subprocess.run(
        [
            'firectl',
            'deploy',
            fine_tuned_model_id,
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)    
    print(result.stderr)  


@app.command()
def get_model(
    fine_tuned_model_id: Annotated[
        str,
        typer.Argument(
            help="ID of a 🛠️ 🦾📈 Fine-tuned 🤖 Model",
            metavar="ID of a 🛠️ 🦾📈 Fine-tuned 🤖 Model",
        )
    ],
    ):
    """
    Get the details of a 🛠️ 🦾📈 Fine-tuned 🤖 Model
    """
    result = subprocess.run(
        [
            'firectl',
            'get',
            'model',
            fine_tuned_model_id,
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)    
    print(result.stderr)  



@app.command()
def mixtral_chat(
    content: Annotated[
        str,
        typer.Argument(
            help="💬 Content to 🗣️  Say to the 🤖 Model",
            metavar="💬 Content for the 🤖 Model",
        )
    ] = "Tell me a joke"
    ):
    """
    🗣️ 💬 Chat with Ⓜ️  Mixtral 8x7B Instruct 
    """
    response = client.chat.completions.create(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        messages=[{
            "role": "user",
            "content": content, 
        }],
    )
    print(response.choices[0].message.content)


@app.command()
def mixtral_code(
    command: Annotated[
        str,
        typer.Argument(
            help="💬 Command to 🗣️  Say to the 🤖 Model",
            metavar="💬 Command for the 🤖 Model",
        )
    ] = "Generate a custom navigation react component with home, blog, store and shopping cart navigation items using typescript and the shadcn/ui library",
    code_only: bool = False,
    comment_code: bool = False,
    verbose: bool = False,
    ):
    """
    💬 Ask Ⓜ️  Mixtral 8x7B Instruct ✍️ write 📝 code
    """
    
    if code_only:
        command = f"{command} and only generate the code in the output"

    if comment_code:
        if verbose:
            command = f"{command} and annotate the code with verbose and explanative comments"
        else:
            command = f"{command} and annotate the code with concise but explanative comments"

    response = client.chat.completions.create(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        messages=[{
            "role": "user",
            "content": command, 
        }],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    app()
