import os, sys

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
    🗣️ 💬 Ask 🦙 Llama 3.1 8B Instruct to ✍️ write 📝 code
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
            help="💬 Input Code to 🗣️  Say to the 🤖 Model",
            metavar="💬 Input Code for the 🤖 Model",
        )
    ] = sys.stdin,
    language: str = "TypeScript and possibly Functional React Components or possibly React Server Components or possibly React Server Actions",
    comment_code: bool = False,
    verbose: bool = False,
    ):
    """
    🗣️ 💬 Ask 🦙 Llama 3.1 8B Instruct to ✍️ write 📝 code
    """
    """Read data from stdin or a file."""
    input_code = input_file.read()

    prompt = ("Read over the following code and look for any errors "
        "or examples of poor coding practices. Then output the code "
        f"with helpful, inline {language} comments added on how to " 
        f"any improve problematic lines of code. \n{input_code}")
    
    print(prompt)
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
    model: Annotated[
        str,
        typer.Argument()
    ], 
    with_data: str = None,
    with_webpage: str = None,
    ):
    """
    🛠️ 🦾📈 Fine-tune a 🤖 Model with 💽 data in a 💾 file or a 🌐📄 webpage 
    """
    if with_data:
        print(f"Okay, which file?")
    if with_webpage:
        print(f"Okay, which page?")


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
    🗣️ 💬 Chat with Ⓜ️  Mixtral 8x7B Instruct 
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
