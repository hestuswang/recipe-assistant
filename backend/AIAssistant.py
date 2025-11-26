import argparse
import subprocess
import pandas as pd
from openai import OpenAI
from dashscope import Application


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="AI Assistant for generating vegetable dishes recommendations.")
    parser.add_argument('--input', type=str, required=True, help="User input for dishes recommendation")
    return parser.parse_args()


# Function to modify input based on existing long-term memory data
def modify_input(user_input):
    # Read the existing data from Excel file
    data = pd.read_excel("LongTermMemory.xlsx", index_col=0)

    # Prepare the long-term memory and user input prompt
    prompt_LongtermMemory = "The content below is information about long term memory\n"
    prompt_UserInput = "The content below is user's input\n"
    input_message = prompt_LongtermMemory

    # Loop through the rows of the data and add them to the prompt
    for i in range(0, data.shape[0]):  # Loop through rows (axis 0), not columns (axis 1)
        input_message += str(data.iloc[i, 0]) + "\n"

    # Add the user's input to the prompt
    input_message += prompt_UserInput + "\n"
    input_message += user_input + "\n"

    return input_message


# Main function to execute all processes
def main():
    # Parse command-line arguments
    args = parse_args()
    user_input = args.input  # Get user input from the command line

    # Start the subprocess in parallel to add the answer to the long-term memory
    process_addLongTerm = subprocess.Popen(
        ['python', 'CreateLongTermMemory.py', '--input', user_input],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # First application: Modify the input for better retrieval in the knowledge base
    client = OpenAI(
        api_key="sk-55f91aaf5605438997475098330d3ab5",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = Application.call(
        api_key="sk-55f91aaf5605438997475098330d3ab5",
        app_id='85e3ba9a2fe44a49ad469e0b640f9c58',
        prompt=modify_input(user_input)
    )
    print(response.output.text)
    print("Our System has finished input modification.")

    # Second application: Get the dishes list
    conscious_conclusion = response.output.text
    response_retrival = Application.call(
        api_key="sk-55f91aaf5605438997475098330d3ab5",
        app_id='d21c0471a54b41fd98acbd06b5f98bf9',
        prompt=conscious_conclusion
    )
    recipes_retrival = response_retrival.output.text
    max_context_round = 3
    if_context = True

    print(response_retrival.output.text)
    print("Our System has finished dish retrieval.")

    # The last application: Detailed help for cooking the dishes
    while max_context_round > 0 and if_context:
        response_helper = Application.call(
            api_key="sk-55f91aaf5605438997475098330d3ab5",
            app_id='33610b43e4d642e49402e49d46f0c2b9',
            prompt=recipes_retrival
        )
        max_context_round -= 1
        print(response_helper.output.text)

        # Ask the user if they want to continue or end the conversation
        keep_going = input(
            "The recommendation has been generated. If you want to continue the conversation, press Y; otherwise, press N.\n")
        if keep_going == "N":
            if_context = False
        else:
            recipes_retrival = input("Please enter your question here:\n")

    # Wait for the subprocess to finish before proceeding
    process_addLongTerm.wait()
    print("The command has been ended.")


# Entry point for the script
if __name__ == "__main__":
    main()
