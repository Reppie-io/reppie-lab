## reppie-labs

Welcome to the reppie-labs repository! This repository is dedicated to showcasing real-world use cases of Generative AI across various industries. Here, you will find examples and demonstrations of how Generative AI technologies are being applied to solve problems and innovate in different sectors.


### How this repository is organized

This repository is organized based on different use cases and industries where Generative AI is making an impact. Each use case is categorized under its respective industry to provide clarity and ease of navigation. The structure of the repository is as follows:

```
reppie-labs/
│
|── libs/ # common libs shared between demos.
|
├── search/
│   ├── ecommerce/
│   └── media/
│
├── summary/
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

We are continuously adding new use cases and industries into this repository to provide a comprehensive overview of Generative AI applications. However, if you are interested in exploring a specific use case or industry that is not yet included in the repository, feel free to reach out to us: contato@reppie.io. 

We are open to collaboration and eager to discuss how we can assist you in exploring the potential of Generative AI for your specific needs.

### Setup Instructions

#### Docker Installation
To run the demonstrations and examples in this repository, Docker is used for containerization, ensuring consistent environments across different systems. Follow the [instructions](https://docs.docker.com/get-docker/) provided in the official Docker documentation.

1. **Linux**: Install Docker Desktop on [Linux](https://docs.docker.com/desktop/install/linux-install/).
2. **Windows**: Install Docker Desktop on [Windows](https://docs.docker.com/desktop/install/windows-install/).
3. **macOS**: Install Docker Desktop fon [Mac](https://docs.docker.com/desktop/install/mac-install/).

Once Docker is installed, ensure that it's running properly on your system before proceeding.

#### Pinecone Setup
Pinecone serves as the vector store for the demonstrations in this repository. Follow these steps to obtain an API key:

1. **Sign Up**: If you haven't already, sign up for a Pinecone account [here](https://app.pinecone.io/?sessionType=signup).
2. **Dashboard**: Log in to your Pinecone account and navigate to the dashboard.
3. **API Key**: In the dashboard, locate the section for API keys. Generate a new API key if you don't have one already.
4. **Copy Key**: Once generated, copy your API key. This key will be used to authenticate your requests to the Pinecone service.

⚠️ Ensure that you keep your Pinecone API key secure and do not expose it in public repositories or insecure environments.

With Docker installed and your Pinecone API key ready, you're all set to explore the examples and demonstrations in the reppie-labs repository. 

If you encounter any issues during setup or have questions about the process, feel free to reach out for assistance: contato@reppie.io. Happy exploring!