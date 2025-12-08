// file: index.mjs
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, QueryCommand } from "@aws-sdk/lib-dynamodb";

const client = new DynamoDBClient({});
const ddb = DynamoDBDocumentClient.from(client);

const TABLE_NAME = "detections";

const headers = {
  "Content-Type": "application/json",
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type,Authorization",
  "Access-Control-Allow-Methods": "GET,OPTIONS",
};

export const handler = async (event) => {
  console.log("Incoming event:", JSON.stringify(event));

  try {
    const qs = event?.queryStringParameters ?? {};

    // how many items to return
    const limit = qs.limit ? Number(qs.limit) : 50;

    // partition key (e.g. camera_id, device_id, etc.)
    // adjust the param name & attr name to your schema
    const partitionKeyValue = 'device';
    if (!partitionKeyValue) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ message: "Missing partition key (e.g. camera_id)" }),
      };
    }

    const exclusiveStartKey = qs.startKey
      ? JSON.parse(decodeURIComponent(qs.startKey))
      : undefined;

    const params = {
      TableName: TABLE_NAME,
      KeyConditionExpression: "#pk = :pk",
      ExpressionAttributeNames: {
        "#pk": "type", // <- change to your real partition key name
      },
      ExpressionAttributeValues: {
        ":pk": partitionKeyValue,
      },
      // false = descending order by sort key (created_at)
      ScanIndexForward: false,
      Limit: limit,
      ExclusiveStartKey: exclusiveStartKey,
    };

    const result = await ddb.send(new QueryCommand(params));

    const responseBody = {
      items: result.Items ?? [],
      lastEvaluatedKey: result.LastEvaluatedKey ?? null,
    };

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(responseBody),
    };
  } catch (err) {
    console.error("Error querying DynamoDB:", err);

    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        message: "Error listing items",
        error: err.message ?? "Unknown error",
      }),
    };
  }
};
