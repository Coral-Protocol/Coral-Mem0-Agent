## [Mem0 Agent](https://github.com/Coral-Protocol/Coral-Mem0-Agent)
 
AI Content Generation Agent with advanced memory capabilities using Mem0 for personalized Reddit post creation. This agent learns from user interactions and preferences to generate increasingly tailored content over time.

## Responsibility
Mem0 Agent serves as an AI Reddit Content Specialist that generates engaging Reddit posts based on user's personality and preferences. It leverages Mem0's memory capabilities to store user requests, learn from interactions, and search through historical context to create personalized content that matches individual user styles and preferences. The agent continuously improves its understanding of each user through persistent memory storage and retrieval.

## Details
- **Framework**: CrewAI
- **Tools used**: Coral MCP Tools, store_user_request_in_mem0, search_mem0_memories
- **AI model**: OpenRouter GPT-4.1-mini
- **Date added**: September 12, 2025
- **License**: MIT

## Setup the Agent

### 1. Clone & Install Dependencies

<details>  

```bash
# In a new terminal clone the repository:
git clone https://github.com/Coral-Protocol/Coral-Mem0-Agent.git

# Navigate to the project directory:
cd Coral-Mem0-Agent

# Download and run the UV installer, setting the installation directory to the current one
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$(pwd) sh

# Create a virtual environment named `.venv` using UV
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate

# install uv
pip install uv

# Install dependencies from `pyproject.toml` using `uv`:
uv sync
```

</details>

### 2. Configure Environment Variables

Get the API Keys:
- [OpenRouter](https://openrouter.ai/keys)
- [MEM0_API_KEY](https://app.mem0.ai/dashboard/api-keys)

<details>

```bash
# Create .env file in project root
cp -r .env.example .env
```
</details>

## Run the Agent

You can run in either of the below modes to get your system running.  

- The Executable Model is part of the Coral Protocol Orchestrator which works with [Coral Studio UI](https://github.com/Coral-Protocol/coral-studio).  
- The Dev Mode allows the Coral Server and all agents to be seperately running on each terminal without UI support.  

### 1. Executable Mode

Checkout: [How to Build a Multi-Agent System with Awesome Open Source Agents using Coral Protocol](https://github.com/Coral-Protocol/existing-agent-sessions-tutorial-private-temp) and update the file: `coral-server/src/main/resources/application.yaml` with the details below, then run the [Coral Server](https://github.com/Coral-Protocol/coral-server) and [Coral Studio UI](https://github.com/Coral-Protocol/coral-studio). You do not need to set up the `.env` in the project directory for running in this mode; it will be captured through the variables below.

<details>

For Linux or MAC:

```bash
# PROJECT_DIR="/PATH/TO/YOUR/PROJECT"

applications:
  - id: "app"
    name: "Default Application"
    description: "Default application for testing"
    privacyKeys:
      - "default-key"
      - "public"
      - "priv"

registry:
  mem0_agent:
    options:
      - name: "OPENROUTER_API_KEY"
        type: "string"
        description: "OpenRouter API key for the service"
      - name: "MEM0_API_KEY"
        type: "string"
        description: "Mem0 API key for memory management"
    runtime:
      type: "executable"
      command: ["bash", "-c", "${PROJECT_DIR}/run_agent.sh main.py"]
      environment:
        - name: "OPENROUTER_API_KEY"
          from: "OPENROUTER_API_KEY"
        - name: "MEM0_API_KEY"
          from: "MEM0_API_KEY"

```

For Windows, create a powershell command (run_agent.ps1) and run:

```bash
command: ["powershell","-ExecutionPolicy", "Bypass", "-File", "${PROJECT_DIR}/run_agent.ps1","main.py"]
```

</details>

### 2. Dev Mode

Ensure that the [Coral Server](https://github.com/Coral-Protocol/coral-server) is running on your system and run below command in a separate terminal.

<details>

```bash
# Run the agent using `uv`:
uv run main.py
```

You can view the agents running in Dev Mode using the [Coral Studio UI](https://github.com/Coral-Protocol/coral-studio) by running it separately in a new terminal.

</details>


## Example

<details>


```bash
# Input:
Interface Agent: Generate me reddit post about ai in healthcare

# Output:
Post 1:
Title: How AI is Revolutionizing Early Disease Detection in Radiology
Content: Recent breakthroughs in computer vision have enabled AI systems to detect early-stage cancers with 94% accuracy, often catching tumors that human radiologists miss. Google's AI can now identify diabetic retinopathy from retinal photographs, potentially preventing blindness in millions of patients worldwide. The key is training on massive datasets of medical images while maintaining patient privacy through federated learning approaches.
Keywords: #AIHealthcare #MachineLearning #MedicalAI #Radiology #EarlyDetection

Post 2:
Title: The Ethics of AI-Powered Drug Discovery: Balancing Innovation with Access
Content: While AI has accelerated drug discovery from 10-15 years to potentially 3-5 years, we're facing new challenges around intellectual property and access. DeepMind's AlphaFold has open-sourced protein structures, but many AI-discovered drugs remain under strict patents. How do we ensure life-saving AI innovations reach those who need them most?
Keywords: #DrugDiscovery #AIEthics #HealthcareAccess #AlphaFold #MedicalInnovation

Post 3:
Title: My Experience Building an AI Chatbot for Mental Health Support
Content: After 6 months of development, our AI mental health companion has helped 10,000+ users manage anxiety and depression. The biggest challenge wasn't the NLP—it was ensuring the AI knew when to recommend human intervention. We implemented safety triggers and 24/7 crisis protocols. Mental health AI isn't about replacing therapists; it's about providing accessible first-line support when humans aren't available.
Keywords: #MentalHealthAI #Chatbots #HealthcareTech #AIForGood #DigitalTherapy

Post 4:
Title: Why Hospital AI Systems Keep Failing (And How to Fix Them)
Content: Deployed 50+ AI models in healthcare settings. 80% failed within 6 months. The problem isn't the algorithms—it's integration with existing workflows. Doctors won't use systems that add 10 minutes to each patient visit. Successful healthcare AI must be invisible, accurate, and faster than current processes. Focus on augmentation, not replacement.
Keywords: #HealthcareAI #DigitalTransformation #MedicalTechnology #AIImplementation #HealthTech

Post 5:
Title: AI Predicting Heart Attacks 5 Years Before They Happen—Here's How
Content: New research shows AI can analyze standard ECGs to predict future cardiac events with 85% accuracy. The model identifies subtle patterns invisible to cardiologists, potentially saving millions of lives through early intervention. The breakthrough came from training on 1.6 million ECGs combined with long-term patient outcomes. Preventive medicine just got a major upgrade.
Keywords: #CardiacAI #PredictiveMedicine #HeartHealth #MedicalAI #PreventiveCare

Note: This agent learns from your preferences and will personalize future content based on your interaction history and stored memories.
```
</details>


## Creator Details
- **Name**: Ahsen Tahir
- **Affiliation**: Coral Protocol
- **Contact**: [Discord](https://discord.com/invite/Xjm892dtt3)