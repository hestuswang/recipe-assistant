import argparse
import pandas as pd
from openai import OpenAI
from dashscope import Application


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Process user input for long-term memory.")
    parser.add_argument('--input', type=str, required=True, help="User input for long-term memory")
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()
    user_input = args.input  # Use the input passed from the command line

    # Read the existing data from Excel file
    data = pd.read_excel("LongTermMemory.xlsx", index_col=0)

    # Prepare the long-term memory and user input prompt
    prompt_LongtermMemory = "The content below is information about long term memory\n"
    prompt_UserInput = "The content below is user's input\n"
    input_message = prompt_LongtermMemory

    # Loop through the rows of the data and add them to the prompt
    for i in range(0, data.shape[0]):  # Fix: should loop through rows (axis 0) not columns (axis 1)
        input_message += str(data.iloc[i, 0]) + "\n"

    # Add the user's input to the prompt
    input_message += prompt_UserInput + "\n"
    input_message += user_input + "\n"

    # Create client (make sure to use your correct API key)
    client = OpenAI(api_key="sk-55f91aaf5605438997475098330d3ab5",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # Make the API call to get the response
    response = Application.call(
        api_key="sk-55f91aaf5605438997475098330d3ab5",
        app_id='83be1f05abb74621aa65a21656641156',
        prompt=input_message
    )

    # Get the output from the response
    answer = response.output.text

    # If the response does not contain "None", add it to the long-term memory (Excel file)
    if "***None***" not in answer:
        new_row = pd.DataFrame({answer})
        data = pd.concat([data, new_row], ignore_index=True, axis=0)
        data.to_excel("LongTermMemory.xlsx")


# Entry point for the script
if __name__ == "__main__":
    main()