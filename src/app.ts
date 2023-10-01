import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { RetrievalQAChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";

import * as dotenv from "dotenv";

dotenv.config();

const loader = new CSVLoader("src/documents/constituents-financials_csv.csv");

const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});

// chunks are created
const splittedDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY
});

const vectorStore = await HNSWLib.fromDocuments(
    splittedDocs,
    embeddings
);

// init LLM MOdel 
const model = new OpenAI({
    modelName: 'gpt-3.5-turbo'
});

const vectorStoreRetriever = vectorStore.asRetriever();

const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);

const question = "What is the EBITDA for Advance Auto Parts?";

const answer = await chain.call({
    query: question
});

console.log({
    question, 
    answer
});

const question1 = "What is the dividend yield for Advance Auto Parts?";

const answer1 = await chain.call({
    query: question1
});

console.log({
    question: question1, 
    answer: answer1
});




