import { DynamoDBClient, PutItemCommand } from "@aws-sdk/client-dynamodb";
const client = new DynamoDBClient({});

const headers = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST,OPTIONS",
  "Access-Control-Allow-Headers": "content-type,authorization",
  "Content-Type": "application/json"
};


export const handler = async (event) => {
  try {
    if (event?.requestContext?.http?.method === "OPTIONS") {
      return { statusCode: 204, headers, body: "" };
    }
    console.log("EVENT 👉", JSON.stringify(event));
  
    const bodyText = event.body || ""; // raw body as string
    console.log("BODY TEXT 👉", bodyText);
  
    let body = {};
    if (bodyText) {
      try {
        body = JSON.parse(bodyText);
      } catch (e) {
        console.log("JSON parse error, keeping raw text only:", e.message);
      }
    }
  
    const now = new Date().toISOString();
    const id = `${now}-${Math.random().toString(36).slice(2, 8)}`;
  
    const item = {
      type: { S: "device" },
      id: { S: id },
      created_at: { S: now },
      raw: { S: bodyText || JSON.stringify(body) || "{}" },
    };
  
    await client.send(
      new PutItemCommand({
        TableName: "detections",
        Item: item,
      })
    );
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ ok: true, id }),
    };
  } catch (e) {
    console.log("ERROR 👉", e.message);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: e.message }),
    };
  }
};
