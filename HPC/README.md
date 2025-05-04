## 🚀 How to Run CUDA Programs in Google Colab

> _Reference: [@shubham0204](https://github.com/shubham0204/PICT_Coursework/blob/lp-5/04_CUDA/CUDA_Compile_Commands.ipynb)_

Follow these steps to compile and execute CUDA (`.cu`) code in Google Colab:

1. **Enable GPU support**
   - Go to **`Runtime > Change runtime type`**
   - Set **Hardware accelerator** to **GPU (preferably T4)**

2. **Save your CUDA code to a file**
   - Use the `%%writefile` command in a new code cell & paste your code:
     ```cpp
     %%writefile file_name.cu
     // paste your CUDA code here
     ```

3. **Verify GPU availability**
   - Run the following to check GPU device details:
     ```bash
     !nvidia-smi
     ```

4. **Compile the CUDA code**
   - Use `nvcc` to compile:
     ```bash
     !nvcc -arch=sm_75 file_name.cu
     ```

5. **Execute the compiled binary**
   - Run the compiled CUDA program:
     ```bash
     !./a.out
     ```
