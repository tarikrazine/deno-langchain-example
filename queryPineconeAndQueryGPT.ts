// 1. Import required modules
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
import { PineconeClient } from "@pinecone-database";

import { load } from "dotenv";

const env = await load();

const openAIApiKey = env["OPENAI_API_KEY"];

// 2. Export the queryPineconeVectorStoreAndQueryLLM function
export const queryPineconeVectorStoreAndQueryLLM = async (
  client: PineconeClient,
  indexName: string,
  question: string,
) => {
  // 3. Start query process
  console.log("Querying Pinecone vector store...");

  // 4. Retrieve the Pinecone index
  const index = client.Index(indexName);

  // 5. Create query embedding
  const queryEmbedding = await new OpenAIEmbeddings({
    openAIApiKey,
  }).embedQuery(question);

  // 6. Query Pinecone index and return top 10 matches
  const queryResponse = await index.query({
    queryRequest: {
      topK: 10,
      vector: queryEmbedding,
      includeMetadata: true,
      includeValues: true,
    },
  });

  if (queryResponse.matches && queryResponse.matches.length) {
    // 7. Log the number of matches
    console.log(`Found ${queryResponse.matches.length} matches...`);
    // 8. Log the question being asked
    console.log(`Asking question: ${question}...`);

    // 9. Create an OpenAI instance and load the QAStuffChain
    const llm = new OpenAI({ openAIApiKey });
    const chain = loadQAStuffChain(llm);

    // 10. Extract and concatenate page content from matched documents
    const concatenatedPageContent = queryResponse.matches
      .map((match: any) => match.metadata && match.metadata.pageContent)
      .join(" ");

    // 11. Execute the chain with input documents and question
    const result = await chain.call({
      input_documents: [new Document({ pageContent: concatenatedPageContent })],
      question: question,
    });

    // 12. Log the answer
    console.log(`Answer: ${result.text}`);
  } else {
    // 13. Log that there are no matches, so GPT-3 will not be queried
    console.log("Since there are no matches, GPT-3 will not be queried.");
  }
};
