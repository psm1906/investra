# Investra - AI-Powered Investment Risk Analysis Platform

## Overview
Investra is a sophisticated investment risk analysis platform that leverages artificial intelligence to provide real-time risk assessments and recommendations for property investments. Built for the WEHack 2025 Capital One Track Challenge, this platform combines advanced machine learning models with user-friendly interfaces to help investors make informed decisions.

## Features
- **Real-time Risk Analysis**: Instant assessment of investment risks using AI models
- **Property Analysis**: Detailed analysis of property investments with multiple risk factors
- **AI Recommendations**: Smart suggestions based on historical data and market trends
- **Interactive Dashboard**: User-friendly interface for visualizing risk metrics
- **Authentication**: Secure user authentication using Clerk
- **Responsive Design**: Modern UI that works across all devices

## Tech Stack
### Frontend
- React with TypeScript
- Vite for build tooling
- Tailwind CSS for styling
- Clerk for authentication
- React Router for navigation

### Backend
- Flask
- Python 3.9+
- Transformers for AI models
- Sentence Transformers for text analysis
- Google Generative AI integration

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/psm1906/investra.git
cd investra
```

2. Install frontend dependencies:
```bash
cd frontend/react
npm install
```

3. Install backend dependencies:
```bash
cd models
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the frontend/react directory with:
```
VITE_CLERK_PUBLISHABLE_KEY=your_clerk_key_here
```

### Running the Application

1. Start the frontend development server:
```bash
cd frontend/react
npm run dev
```

2. Start the backend server:
```bash
cd models
python app.py
```

The application will be available at `http://localhost:5173`

## Project Structure
```
investra/
├── frontend/
│   └── react/
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   └── assets/
│       └── public/
├── models/
│   ├── app.py
│   └── requirements.txt
└── README.md
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments
- WEHack 2025 for the platform
- Capital One for the challenge
- All contributors and supporters of the project
