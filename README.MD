# Deploying Flask Applications on Google Cloud Run: A Step-by-Step Guide

This repository contains a comprehensive tutorial designed to help beginners deploy their Flask applications using Docker and Google Cloud Run. The tutorial provides a detailed, step-by-step guide on how to containerize a Flask application, deploy it to Google Cloud Run, and manage it effectively.

**What You Will Learn:**
- How to create a Dockerfile and build a Docker image for a Flask application.
- Steps to push the Docker image to Google Container Registry (GCR).
- How to deploy the containerized application to Google Cloud Run, allowing it to scale automatically and handle requests efficiently.
- Methods to test the deployment and ensure your application is functioning as expected.
- Techniques to monitor and manage the deployed application using Google Cloud Console.

**Features of This Tutorial:**
- **Beginner-Friendly:** Designed with beginners in mind to help them understand each step thoroughly.
- **Hands-On Approach:** Includes practical, executable commands and a real-world application example.
- **Focus on Best Practices:** Emphasizes best practices in cloud deployment and Docker usage.

Whether you are a student, a hobbyist, or a professional developer starting with cloud deployments, this tutorial will provide you with the necessary skills to deploy and manage your web applications on Google Cloud Run efficiently.

Feel free to clone, fork, and explore the tutorial. Contributions to improve the tutorial or add more features are welcome!



## Step 1: Prepare Your Model
### Requirements
The project dependencies are listed in the requirements.txt file. Here's a quick overview:

```shell
torch>=1.12.0
torchvision>=0.13.0  # Optional, if needed for image processing
scikit-learn>=1.1.1
Flask
```


#### Create a PyTorch Model by running iris_cls_train.py
```shell
python iris_cls_train.py
```






## Step 2: Create a Google Cloud Account
- Go to the [Google Cloud](https://cloud.google.com/) website and sign up for an account.
- Set up a billing account (note that Google offers a free trial with $300 credit).
Enable the Required Services
You'll need to enable several services on GCP, including Google Cloud Build and Cloud Run, which are essential for building and deploying your application.

#### Open the Google [Cloud Console](https://console.cloud.google.com/).
- Navigate to "APIs & Services" on the left-hand menu.
- Enable APIs and Services by searching for and enabling the following:
  - Cloud Run API
  - Cloud Build API
  - Container Registry API



## Step 3: Download the Google Cloud SDK
#### Download the Google Cloud SDK
Go to the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) webpage: Visit Google Cloud SDK to find the latest version of the SDK.
- Choose the correct version for your operating system (Windows, macOS, or Linux).
- Download the installer or the archive as appropriate for your system.
- Follow the installation instructions for your operating system.

#### Initialize the Google Cloud SDK
- Open your command line or terminal.
- Run the initialization command:
```shell
gcloud init
```
- Log in to your Google account when prompted by the SDK. This will open a new browser window asking you to choose a Google account and grant the Google Cloud SDK permission to access your Google Cloud resources.
- Select a Cloud Project: After logging in, the SDK will ask you to select a project from your GCP account or create a new one.

#### Verify Installation
After initializing, verify that the installation and setup were successful by running:
```shell
gcloud info
```

This command displays information about your current environment, including the active account and project.

#### List the Projects:
Run the following command to see all the projects associated with your current Google Cloud account:
    ```shell
    gcloud projects list
    ```
#### Set This Project as Your Default:
To simplify future commands, set this as your default project in your Google Cloud SDK settings. Run the following command in your Command Prompt:

```shell
  gcloud config set project buoyant-program-442017-k3
```
This command will configure your Google Cloud SDK to automatically use this project for all operations, eliminating the need to specify the project ID in future commands.

#### Set Compute Zone:
Choose a default compute zone, which is the region where your resources will be located by default. For example, us-central1-a, for USA, or europe-west2-a, for Europe. Run the following command to set the default compute zone:

```shell
gcloud config set compute/zone europe-west2-a
```

 Now your Google Cloud environment is ready to deploy applications, create resources, and make the most out of the Google Cloud services available to you.



## Step 5: Install Necessary Tools and Set Up Your Environment
#### Install Docker:
You'll need Docker to create containers for your model. You can download and install Docker from [Docker](https://docs.docker.com/desktop/) official site.

Docker Desktop is the tool you'll need to create, manage, and run Docker containers on your Windows system. It provides a graphical user interface (GUI), making it easier to manage containers and Docker resources.

- Download Docker Desktop for Windows:
Visit the Docker Hub and download the Docker Desktop installer for Windows.

- Install Docker Desktop:
Run the installer you downloaded. Follow the on-screen instructions to complete the installation. This may include enabling the WSL 2 feature on your Windows and possibly restarting your machine.

- Verify Installation:
Once installed, you can open the Docker Desktop application to ensure it's running correctly. You can also open your command prompt and type ```docker --version``` to see if Docker responds with the version number, indicating it’s correctly installed.

    ```shell
    docker --version
    ```


## Step 6: Prepare Your Flask Application

Your Flask application will serve as the interface for your PyTorch model predictions. Below are the steps to prepare it:

1. **Create Flask Application (`app.py`)**:
   - Set up your Flask app using the following template. Adjust the path to your PyTorch model file (`iris_model.pt`) as necessary.

   ```python
   from flask import Flask, request, jsonify
   import torch

   app = Flask(__name__)

   # Load your trained TorchScript model
   model = torch.jit.load('iris_model.pt')

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.get_json()
       inputs = torch.tensor(data['inputs'], dtype=torch.float32)
       with torch.no_grad():
           prediction = model(inputs)
       return jsonify({'prediction': prediction.tolist()})

   if __name__ == '__main__':
       app.run(debug=True, host='0.0.0.0', port=8080)
    ```
2. **Create requirements.txt:**
- Create a file named requirements.txt in your project directory with the following contents to specify the necessary Python packages:
    
    ```shell
    Flask
    torch>=1.12.0
    ```
  2. **Test Your Flask Application Locally:**
  Before containerizing, you can test the Flask application locally. This will start a local web server. You can use [Postman](https://www.postman.com/downloads/) (useful if user needs a GUI) or ```curl``` (useful when running from CMD) to send a POST request to ```http://localhost:8080/predict``` with JSON data to test if it returns predictions correctly:

     - Run python app.py in your command prompt to start a local web server (this to be done from the same env, e.g., conda env, where the dependencies are installed):

         ```shell
         python app.py
         ```

     - Test the server using Postman or curl by sending a POST request to ```http://localhost:8080/predict``` with appropriate JSON data to ensure it returns predictions accurately. Example curl command (this is to be done from a separate terminal window, no need to be in the same directory or env as the Flask app):

         ```shell
         curl -X POST -H "Content-Type: application/json" -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}' http://localhost:8080/predict
         ```
       or this if testing from Windows CMD (Windows handles single quotes differently):
         ```shell
       curl -X POST -H "Content-Type: application/json" -d "{\"inputs\": [[5.1, 3.5, 1.4, 0.2]]}" http://localhost:8080/predict
        ```



Once Docker is set up and your Flask application is ready and tested locally, you can proceed with containerization, involving creating a Dockerfile and building a Docker image. 


## Step 7: Containerize Your Application

Before deploying to Google Cloud Run, containerize your application using Docker. This involves creating a Dockerfile, building a Docker image, and pushing that image to Google Container Registry (GCR).

### Create a Dockerfile

Create a `Dockerfile` in the root of your project directory with the following content:

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```


This ```Dockerfile``` sets up a Python environment, installs your dependencies, and sets the command to run your Flask application. **Make sure Docker is running** on your system before proceeding.

Build Your Docker Image
Open your terminal, navigate to your project directory where your ```Dockerfile``` is, and run:

```shell
docker build -t gcr.io/your-project-id/your-app-name .
```
Replace ```your-project-id``` with your Google Cloud project ID and ```your-app-name``` with a name for your app. Use ```gcoud projects list``` to find your project ID. Mind that our-app-name cannot have spaces, and is not directly related to the name of the Dockerfile or the Flask app. It's a name you choose for your container image.


Push the Docker Image to Google Container Registry
- First, ensure you're authenticated with Google Cloud:

    ```shell
    gcloud auth configure-docker
    ```

- Then, push your Docker image:

    ```shell
    docker push gcr.io/your-project-id/your-app-name
    ```
This prepares your application for deployment to Google Cloud Run by creating a container image that can be managed by Google's infrastructure.






