
# Straw Hat Coding Assistant 🤖

## Project Overview

The **Straw Hat Coding Assistant** is an AI-powered coding assistant built on the fine-tuned **Llama 3.1 8B** model. This project leverages advanced machine learning techniques to enhance coding assistance through a user-friendly interface. The Llama 3.1 8B model has been fine-tuned specifically to understand and generate code-related responses, making it a valuable tool for developers and learners alike. 🚀

## Model Description

The Llama 3.1 8B model is a state-of-the-art transformer-based language model designed for a variety of natural language processing tasks. With 8 billion parameters, it excels in generating human-like text, understanding context, and providing accurate code suggestions. This project fine-tunes the model to optimize its performance in coding assistance, ensuring that it aligns closely with user preferences through Direct Preference Optimization (DPO).

## Fine-Tuning Process 🔧

The fine-tuning process involves training the Llama 3.1 8B model on a specific dataset tailored for coding tasks. This process adjusts the model's weights to improve its ability to generate relevant and accurate code suggestions based on user inputs. By leveraging supervised fine-tuning techniques, the model learns from examples, enhancing its understanding of programming languages and best practices.

## Direct Preference Optimization (DPO) 🎯

Direct Preference Optimization (DPO) is a technique used to align the model's outputs with user preferences. After the initial fine-tuning, DPO further refines the model by incorporating user feedback and preferences into the training process. This ensures that the model not only generates accurate code but also aligns closely with the specific needs and expectations of its users, resulting in a more personalized coding assistant experience.

## Project Structure 📁

- `src/fine_tuning.py`: Code for fine-tuning the model using a dataset.
- `src/dpo_training.py`: Code for applying Direct Preference Optimization.
- `src/streamlit_app.py`: A Streamlit application that interacts with the model.


## Steps to Run the Project 🏃‍♂️

### 1. Installation

- **Clone the Repository**:
  ```bash
  git clone https://github.com/VivekChauhan05/Straw-Hat-Llama3.1-8B-Finetuning-DPO
  cd Straw-Hat-Llama3.1-8B-Finetuning-DPO
  ```

- **Install Required Packages**:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Fine-Tuning the Model

- **Run the Fine-Tuning Script**:
  Execute the following command to fine-tune the model:

  ```bash
  python src/fine_tuning.py
  ```

### 3. Direct Preference Optimization (DPO) Training

- **Execute the DPO Training Script**:
  After fine-tuning, run the DPO training script:

  ```bash
  python src/dpo_training.py
  ```

### 4. Launch the Streamlit Application

- **Start the Streamlit App**:
  To interact with the model, launch the Streamlit application:

  ```bash
  streamlit run src/app.py
  ```

## Usage 💻

Once the Streamlit app is running, you can input prompts and adjust parameters such as temperature, top-k, and max length to generate coding suggestions. The assistant will provide concise, readable, and well-documented Python code based on your input.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License 📜

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for more details.