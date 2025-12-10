/*
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/
import styles from "@/styles/WelcomeSection.module.css";

interface WelcomeSectionProps {
  setQuery: (value: string) => void;
}

export default function WelcomeSection({ setQuery }: WelcomeSectionProps) {
  const promptTemplates = {
    rag: "What is the Blackwell GB202 GPU according to the whitepaper document I uploaded?",
    imageGen:
      "Generate an Image: Create a cartoon image of a squirrel holding an elephant that is holding a weasel that is holding a beetle that is holding an ant that is holding a cupcake.",
    videoGen:
      "Generate a video: Large robot throwing a a smaller robot that is carrying an even smaller robot that is wearing a dog costume.",
    imageRecog:
      "Image Recognition: Describe this uploaded picture and extract any safety-relevant details.",
    videoRecog:
      "Video Recognition: Summarize the main actions and visual details in this clip.",
    code: `Can you generate code to develop a responsive personal website for my freelance AI dev business based on my personal brand palette?

My palette is:
#606C38
#283618
#FEFAE0
#DDA15E
#BC6C25`
  };

  const handleCardClick = (promptKey: keyof typeof promptTemplates) => {
    setQuery(promptTemplates[promptKey]);
  };

  return (
    <div className={styles.welcomeContainer}>
      <div className={styles.welcomeMessage}>
        Hello! Send a message to start chatting with DGX Station.
      </div>
      <div className={styles.agentCards}>
        <div 
          className={`${styles.agentCard} ${styles.animate1}`}
          onClick={() => handleCardClick('rag')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <circle cx="11" cy="11" r="8"/>
              <path d="m21 21-4.35-4.35"/>
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Search Documents</h3>
          <p className={styles.agentSubtitle}>RAG Agent</p>
        </div>
        <div
          className={`${styles.agentCard} ${styles.animate2}`}
          onClick={() => handleCardClick('imageGen')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <path d="M4 5h16v14H4z" />
              <path d="m7 14 3-4 2 3 3-5 2 3" />
              <circle cx="9" cy="8" r="1.25" />
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Generate an Image</h3>
          <p className={styles.agentSubtitle}>FLUX image creation prompt</p>
        </div>
        <div
          className={`${styles.agentCard} ${styles.animate3}`}
          onClick={() => handleCardClick('videoGen')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <rect x="2" y="5" width="15" height="14" rx="2" ry="2" />
              <polygon points="17 8 22 6 22 18 17 16 17 8" />
              <path d="M6 10h5l-2 3 2 3H6" />
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Generate a Video</h3>
          <p className={styles.agentSubtitle}>Wan2.2 text-to-video</p>
        </div>
        <div
          className={`${styles.agentCard} ${styles.animate5}`}
          onClick={() => handleCardClick('imageRecog')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <rect x="3" y="3" width="18" height="14" rx="2" ry="2" />
              <circle cx="8.5" cy="9.5" r="1.5" />
              <path d="M13 11.5 15 9l4 5" />
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Image Recognition</h3>
          <p className={styles.agentSubtitle}>Analyze uploaded photos</p>
        </div>
        <div
          className={`${styles.agentCard} ${styles.animate6}`}
          onClick={() => handleCardClick('videoRecog')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <rect x="2" y="4" width="20" height="14" rx="2" ry="2" />
              <polygon points="10 9 15 12 10 15 10 9" />
              <path d="M2 12h4" />
              <path d="M18 12h4" />
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Video Recognition</h3>
          <p className={styles.agentSubtitle}>Summarize video content</p>
        </div>
        <div
          className={`${styles.agentCard} ${styles.animate4}`}
          onClick={() => handleCardClick('code')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <polyline points="16,18 22,12 16,6"/>
              <polyline points="8,6 2,12 8,18"/>
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Code Generation</h3>
          <p className={styles.agentSubtitle}>Coding Agent</p>
        </div>
      </div>
    </div>
  );
}
