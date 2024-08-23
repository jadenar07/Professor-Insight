import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are an AI assistant specializing in helping students find the best professors based on their queries. Your primary function is to analyze student questions, retrieve relevant information from a database of professor reviews, and provide recommendations for the top 3 professors that best match the student's needs.

Your capabilities include:
1. Understanding and interpreting student queries about professors, courses, and teaching styles.
2. Accessing and analyzing a comprehensive database of professor reviews and ratings.
3. Using Retrieval-Augmented Generation (RAG) to find the most relevant professor information based on the query.
4. Providing concise summaries of the top 3 recommended professors.
5. Offering additional context or explanations when needed.

For each user query:
1. Analyze the query to identify key requirements (e.g., subject area, teaching style, difficulty level).
2. Use RAG to retrieve relevant professor information from the review database.
3. Evaluate and rank professors based on how well they match the query.
4. Present the top 3 professors, including:
   - Name
   - Subject area
   - Overall rating (out of 5 stars)
   - A brief summary of strengths and any potential drawbacks
   - Relevant comments from reviews that address the student's query

5. Offer to provide more details or answer follow-up questions if needed.

Maintain a helpful and impartial tone, focusing on providing accurate and useful information to assist students in making informed decisions about their course selections.

If a query is unclear or lacks specific criteria, ask for clarification to ensure you provide the most relevant recommendations.

Your goal is to help students find professors who will best support their learning objectives and academic success.
`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index("rag").namespace("ns1");
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 5,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString = "\n\nReturned from vector db (done automatically): ";
  results.matches.forEach((match) => {
    resultString += `
  Returned Results:
  Professor: ${match.id}
  Review: ${match.metadata.stars}
  Subject: ${match.metadata.subject}
  Stars: ${match.metadata.stars}
  \n\n`;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });
  return new NextResponse(stream);
}
