import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { RetrievalQAChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
import * as dotenv from "dotenv";

dotenv.config();

const loader = new PDFLoader("src/documents/budget_speech.pdf");

const docs = await loader.load();

// splitter function
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 20,
});

// created chunks from pdf
const splittedDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings();

const vectorStore = await HNSWLib.fromDocuments(
  splittedDocs,
  embeddings
);

const vectorStoreRetriever = vectorStore.asRetriever();
const model = new OpenAI({
  modelName: 'gpt-3.5-turbo'
});

const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);

const question = 'What is the theme of G20?';

const answer = await chain.call({
  query: question
});

console.log({
  question,
  answer
});

const question1 = 'What is the budget allocated for railways?';
const answer1 = await chain.call({
  query: question1
});

console.log({
  question: question1,
  answer: answer1
});

const question2 = 'what is PM Matsya Sampada Yojana?'
const answer2 = await chain.call({
  query: question2
});
console.log({
  question: question2,
  answer: answer2
});



