// Inquiry types — match app/control_plane/inquiries_api.py

export interface InquirySummary {
  filename: string;
  date: string | null;
  slug: string | null;
  question_text: string;
  preview: string;
  size_bytes: number;
  modified_at: string;
}

export interface InquiryDetail extends InquirySummary {
  body: string;
}

export interface InquiryListResponse {
  count: number;
  inquiries: InquirySummary[];
}

export interface InquiryQuestion {
  slug: string;
  text: string;
  framing: string;
  most_recent_answer_date: string | null;
}

export interface InquiryQuestionsResponse {
  count: number;
  questions: InquiryQuestion[];
}
