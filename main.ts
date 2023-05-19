import { PineconeClient } from "@pinecone-database";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { Document } from "langchain/document";
import { load } from "dotenv";

import { createPineconeIndex } from "./createPineconeIndex.ts";
import { updatePinecone } from "./updatePinecone.ts";
import { queryPineconeVectorStoreAndQueryLLM } from "./queryPineconeAndQueryGPT.ts";

// Load environment variables
const env = await load();

// Set up DirectoryLoader to load documents from the ./documents directory
const loader = new DirectoryLoader("./documents", {
  ".txt": (path) => new TextLoader(path),
});

const docs: Document[] = await loader.load();

// Set up variables for the filename, question, and index settings
const question = "Who she id juliet?";
const indexName = "romeo-and-juliet";
const vectorDimension = 1536;

// Initialize Pinecone client with API key and environment
const client = new PineconeClient();
await client.init({
  apiKey: env["PINECONE_API_KEY"],
  environment: env["PINECONE_ENVIRONMENT"],
});

await createPineconeIndex(client, indexName, vectorDimension);

// 12. Update Pinecone vector store with document embeddings
//await updatePinecone(client, indexName, docs);

// 13. Query Pinecone vector store and GPT model for an answer
await queryPineconeVectorStoreAndQueryLLM(client, indexName, question);
