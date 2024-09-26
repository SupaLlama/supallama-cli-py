import os
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
def chat_with_llama(
    content: Annotated[
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
            "content": content, 
        }],
    )
    print(response.choices[0].message.content)


@app.command()
def chat_with_mixtral(
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
    if with_webpage:
        print(f"Okay, which page?")


if __name__ == "__main__":
    app()
