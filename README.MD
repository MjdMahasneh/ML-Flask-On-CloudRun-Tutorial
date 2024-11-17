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




## Step 8: Deploy Your Docker Image to Google Cloud Run
Now that your Docker image is in the Google Container Registry, you can deploy it to Google Cloud Run. This service will handle the infrastructure, allowing your application to automatically scale based on traffic.

In your terminal or command prompt:


```shell
gcloud run deploy SERVICE_NAME --image gcr.io/PROJECT_ID/IMAGE_NAME --platform managed --allow-unauthenticated

## e.g., gcloud run deploy my-first-project --image gcr.io/buoyant-program-442017-k3/my-first-project --platform managed --allow-unauthenticated
```

This command will start the deployment process, and it might ask you to specify the region if you haven't set a default region.

#### Configuration Details

- **`SERVICE_NAME`**: This is the name you assign to your service on Cloud Run. Choose a simple, descriptive name for your service, which will help you identify it within the Google Cloud Console. This name can be different from your Docker image name and should be relevant to your application.

- **`--image gcr.io/PROJECT_ID/IMAGE_NAME`**: Replace `PROJECT_ID` with your Google Cloud project ID and `IMAGE_NAME` with the name of your Docker image stored in Google Container Registry. Ensure the image path correctly points to the image in Google Container Registry. This path includes your project's unique ID and the image name.

- **`--platform managed`**: This option specifies that you are deploying to the fully managed environment of Cloud Run, which handles all infrastructure management tasks. This setting is crucial for specifying that Google should manage the underlying infrastructure, making deployment easier and more scalable.

- **`--allow-unauthenticated`**: This flag allows public access to your application without requiring authentication. By default, Cloud Run services require authentication. This flag makes your application accessible to anyone on the internet.





- **Verify Deployment**
After running the deployment command, Google Cloud Run will output information about the deployment, including the URL where your application is hosted.

Check the provided URL in your browser or use a tool like curl to make sure your application is running correctly.

## Step 9: Monitor and Manage Your Deployment

Once your application is deployed, you can manage and monitor it using the Google Cloud Console.

- Navigate to Google Cloud Run in the Google Cloud Console.
- Select your deployed service to view details, manage settings, see logs, and monitor usage.



## Step 10: Test Your Deployed Application


- **Obtain the Service URL**
    After deploying your application to Google Cloud Run, you will receive a URL where your service is accessible. 
    This URL is provided at the end of the deployment process and can also be found in the Google Cloud Console under the Cloud Run service details.

- **Send Requests to the Application**
    You can test your application by sending HTTP requests to the provided URL using tools like ```curl``` or [Postman](https://www.postman.com/downloads/). This allows you to verify that your application is correctly deployed and responding as expected.

  - **Using curl**
    You can use the curl command from your terminal to send a request to your application. Here’s an example based on the typical use case of a POST request expecting JSON data:
    
    ```shell
    curl -X POST -H "Content-Type: application/json" -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}' YOUR_SERVICE_URL/predict
    ```
    on Windows (open a new CMD window, no need to be in the same directory as the Flask app or Dockerfile):
    ```shell
    curl -X POST -H "Content-Type: application/json" -d "{\"inputs\": [[5.1, 3.5, 1.4, 0.2]]}" https://my-first-project-415802471809.europe-west2.run.app/predict
    ```

    Replace YOUR_SERVICE_URL with the URL provided after deployment. This command sends a sample JSON payload which should be adjusted according to what your application expects.

     - ```X POST```: Specifies the HTTP method as POST.

     - ```-H "Content-Type: application/json"```: Sets the header to indicate that the data being sent is in JSON format.
     - ```-d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}'```: Defines the data being sent to the server. This JSON is structured according to how your Flask app expects input data. Make sure that the JSON structure aligns with what your Flask route ```/predict``` is programmed to parse.

[//]: # (curl -X POST -H "Content-Type: application/json" -d "{\"inputs\": [[5.1, 3.5, 1.4, 0.2]]}" https://my-first-project-415802471809.europe-west2.run.app/predict)


- **Review the Response**

  Check the responses you receive:
  - **Success**: A successful response will typically match what you expect your Flask application to return based on the input.
  - **Error Handling**: If there are errors, the response should ideally contain error messages or codes that help identify the issue.

- **Monitor Application Logs**

    If the initial tests reveal issues or if you want to understand the application's behavior better under different conditions:

  - Go to the Google Cloud Console.
  - Navigate to the Cloud Run section.
  - Select your application and go to the "Logs" tab to view detailed logs of incoming requests and system responses.

- **Automated Testing**

    If you anticipate frequent updates or want to ensure continuous reliability, consider setting up automated tests using tools like GitHub Actions, which can programmatically send requests to your application and verify the responses against expected outcomes.

Testing your deployed application thoroughly will help ensure that it behaves as intended in a live environment, providing confidence in its reliability and functionality. If you encounter any specific issues during testing, you might need to adjust the application's code or configuration and redeploy.



## Next Steps
- **Optimize Configuration**: Depending on your application’s needs, consider setting environment variables, adjusting memory limits, or configuring custom domain mappings through the Google Cloud Console.
- **Continuous Deployment**: Set up continuous deployment using Google Cloud Build or GitHub Actions to automatically deploy updates to your application whenever you push changes to your repository.
- **Secure Your Application**: If needed, change the --allow-unauthenticated flag to restrict access, and use Identity-Aware Proxy or other authentication methods to control access.

By following these steps, your Flask application will be running on a scalable, serverless platform, leveraging Google Cloud Run’s powerful features for easy deployment and management. 


