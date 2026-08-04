import { DynamoDBClient, PutItemCommand } from "@aws-sdk/client-dynamodb";
const client = new DynamoDBClient({});

export const handler = async (event) => {
  console.log("EVENT 👉", JSON.stringify(event, null, 2));

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
    type: { S: "event" },
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
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ok: true, id }),
  };
};
