import { KnowledgeBase, ChatMessage } from './types';

export const mockKnowledgeBases: KnowledgeBase[] = [
  {
    id: '1',
    name: 'Health Insurance Policy',
    icon: '🏥',
    lastUpdated: '2024-01-15',
    documentCount: 12,
    createdAt: '2024-01-10'
  },
  {
    id: '2',
    name: 'Company Handbook',
    icon: '📚',
    lastUpdated: '2024-01-14',
    documentCount: 8,
    createdAt: '2024-01-08'
  },
  {
    id: '3',
    name: 'Technical Documentation',
    icon: '⚙️',
    lastUpdated: '2024-01-13',
    documentCount: 25,
    createdAt: '2024-01-05'
  },
  {
    id: '4',
    name: 'Legal Contracts',
    icon: '⚖️',
    lastUpdated: '2024-01-12',
    documentCount: 15,
    createdAt: '2024-01-03'
  }
];

export const mockChatMessages: Record<string, ChatMessage[]> = {
  '1': [
    {
      id: '1',
      content: 'What is covered under my health insurance policy?',
      type: 'user',
      timestamp: '2024-01-15T10:30:00Z'
    },
    {
      id: '2',
      content: 'Based on your health insurance policy documents, your coverage includes:\n\n• **Medical Services**: Doctor visits, specialist consultations, and emergency care\n• **Hospitalization**: Inpatient care, surgery, and room & board\n• **Prescription Drugs**: Generic and brand-name medications with tier-based copays\n• **Preventive Care**: Annual checkups, vaccinations, and screenings at no cost\n• **Mental Health**: Therapy sessions and psychiatric care\n\nWould you like me to explain any specific coverage area in more detail?',
      type: 'ai',
      timestamp: '2024-01-15T10:30:05Z'
    },
    {
      id: '3',
      content: 'What about dental coverage?',
      type: 'user',
      timestamp: '2024-01-15T10:32:00Z'
    },
    {
      id: '4',
      content: 'According to your policy documents, dental coverage is included with the following benefits:\n\n• **Preventive Care**: Cleanings, exams, and X-rays covered at 100%\n• **Basic Procedures**: Fillings and extractions covered at 80%\n• **Major Procedures**: Crowns, bridges, and root canals covered at 50%\n• **Annual Maximum**: $1,500 per calendar year\n\nNote that there\'s a 6-month waiting period for basic procedures and 12 months for major procedures.',
      type: 'ai',
      timestamp: '2024-01-15T10:32:08Z'
    }
  ]
};