import React from 'react';

interface AiRecommendationProps {
  content: string;
}

const AiRecommendation: React.FC<AiRecommendationProps> = ({ content }) => {
  // Parse the markdown report into sections based on known headers.
  const parseSections = (text: string) => {
    const lines = text.split("\n");
    const sections: { [key: string]: string } = {};
    let currentSection = "";
    // Define the headers we expect.
    const headers = ["Summary:", "View Full Report:", "Pros:", "Cons:", "High Risks:", "Verdict:"];
    
    lines.forEach((line) => {
      const trimmed = line.trim();
      if (headers.includes(trimmed)) {
        // Start a new section; remove the colon for display purposes.
        currentSection = trimmed.slice(0, -1);
        sections[currentSection] = "";
      } else if (currentSection) {
        sections[currentSection] += line + "\n";
      }
    });
    
    // Trim whitespace from each section.
    Object.keys(sections).forEach(key => {
      sections[key] = sections[key].trim();
    });
    
    return sections;
  };

  const sections = parseSections(content);

  // Prepare the summary display:
  // Display the summary content (without "Summary:" header) and append the final verdict.
  let summaryDisplay = "";
  if (sections["Summary"]) {
    summaryDisplay += sections["Summary"];
  }
  if (sections["Verdict"]) {
    summaryDisplay += "\n\nFinal Verdict: " + sections["Verdict"];
  }

  // Download the full report as a Markdown file.
  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'investment_analysis.md';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="ai-summary" style={{ whiteSpace: 'pre-wrap', marginBottom: '1rem' }}>
        {summaryDisplay}
      </div>
      <button onClick={handleDownload}>Download Full Report (Markdown)</button>
    </div>
  );
};

export default AiRecommendation;