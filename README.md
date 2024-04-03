## reppie-labs

Welcome to the **reppie-labs** repository! 

This repository showcases practical examples of how Generative AI is being used to solve problems and drive innovation across various industries. You will find clear demonstrations of real-world Generative AI applications that you can learn from and build upon.


### Repository organization

**reppie-labs** is structured around specific use cases. Each use case folder includes relevant industry examples, making it easy to grasp the code's practical applications.

The repository organization:

```
reppie-labs/
│
|── libs/ # common libs shared between applications.
|
├── search/
│   ├── ecommerce/
│   └── content/
│
├── summarization/
│   ├── sales/
│   └── legal/
│
├── chatbot-assistant/
│   ├── customer-service/
│   └── education/
│
└── image-generation/
│   ├── advertising/
│   └── fashion/
...
```

We are constantly expanding this repository to include new examples and scenarios. If you don't find a specific use case or industry that you're interested in, please don't hesitate to contact us at contato@reppie.io. 

We are open to collaboration and would be delighted to discuss how we can help you explore the potential of Generative AI tailored to your unique requirements.


### Setup Instructions

#### 1. Docker Installation
This repository uses Docker containers to run the demonstrations and examples. This ensures a consistent environment regardless of your system.  
See the official [Docker documentation](https://docs.docker.com/get-docker/) for setup instructions: 

1. **Linux**: Install Docker Desktop on [Linux](https://docs.docker.com/desktop/install/linux-install/).
2. **Windows**: Install Docker Desktop on [Windows](https://docs.docker.com/desktop/install/windows-install/).
3. **macOS**: Install Docker Desktop for [Mac](https://docs.docker.com/desktop/install/mac-install/).

After installing Docker, ensure that it is running properly on your system before moving forward.

#### 2. Pinecone Setup
Pinecone is used as the vector store for the examples in this repository. To [set up Pinecone](https://docs.pinecone.io/docs/quickstart) and obtain an API key, follow these steps:
1. **Login or Sign up**: If you don't have a Pinecone account, sign up for one by visiting the [Pinecone website](https://app.pinecone.io/?sessionType=signup) and completing the registration process.
2. **Get your API Key**: Open the Pinecone Console, go to API Keys, and copy your API key to a secure location. This key will be required to authenticate your requests when interacting with the Pinecone service.

⚠️ Important: Keep your Pinecone API key confidential and avoid exposing it in public repositories or insecure environments. Treat it as sensitive information to maintain the security of your Pinecone account.

#### 3. Creating an OpenAI Account and API Key

Follow these steps to set up your OpenAI account and generate an API key:

1. **Sign Up for OpenAI**: Go to the OpenAI website (https://openai.com/), click on the 'Sign Up' button. Fill in your details and follow the prompts to create an account. Ignore this step if you already have a account.

2. **Create an API Key**: Once logged in, navigate to the API section by clicking on 'API' in the menu. In the API dashboard, click on 'Create new key'. Give your key a name and select the appropriate access and permissions.

3. **Secure Your API Key**: Copy the API key to a secure location; you won't be able to see it again.
Use this key in your application to authenticate with OpenAI's services. Remember to keep your API key confidential to protect your account and services.

With Docker installed and your Pinecone and OpenAI API keys ready, you have everything you need to start exploring the examples and demonstrations provided in the **reppie-labs** repository.
If you have any questions or encounter issues during the setup process, feel free to reach out at contato@reppie.io. 

Happy exploring!
