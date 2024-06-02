import subprocess

# Function to run the executable with different parameters
def run_executable_with_parameters(executable_name, parameters):
    outputs = []
    for params in parameters:
        run_command = [f"./{executable_name}"] + params
        result = subprocess.run(run_command, capture_output=True, text=True)
        if result.returncode != 0:
            outputs.append(f"Error: {result.stderr.strip()}")
        else:
            outputs.append(result.stdout.strip())
    return outputs

# Function to save outputs to a file
def save_outputs_to_file(output_file, parameters, outputs):
    with open(output_file, "w") as file:
        for params, output in zip(parameters, outputs):
            file.write(f"Parameters: {' '.join(params)}\n")
            file.write(f"Output: {output}\n")
            file.write("\n")

if __name__ == "__main__":
    executable_name = "your_executable"  # Name of your executable
    output_file = "outputs.txt"

    # List of solving files (1 10 100 1000 10000)
    parameters = [
        ["../systems/1_sys.txt", "../systems/10_sys.txt"],
        ["../systems/100_sys.txt", "../systems/1000_sys.txt"],
        ["../systems/10000_sys.txt"]
    ]

    outputs = run_executable_with_parameters(executable_name, parameters)
    save_outputs_to_file(output_file, parameters, outputs)
    print(f"Outputs saved to {output_file}")
