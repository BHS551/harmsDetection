"""
create_queues.py — crea (idempotente) las colas SQS entre capas y muestra sus URLs.
  heimdall-candidates : capa0 (movimiento) -> capa1 (clip)
  heimdall-vlm        : capa1 (clip)        -> capa2 (vlm)
Cada cola tiene una DLQ para mensajes que fallan repetidamente.
"""
import json
import boto3

sqs = boto3.client("sqs", region_name="us-east-1")

QUEUES = ["heimdall-candidates", "heimdall-vlm"]


def ensure_queue(name):
    dlq_name = f"{name}-dlq"
    dlq_url = sqs.create_queue(QueueName=dlq_name)["QueueUrl"]
    dlq_arn = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
    url = sqs.create_queue(QueueName=name, Attributes={
        "VisibilityTimeout": "120",           # tiempo para procesar (CLIP/VLM) antes de re-entregar
        "MessageRetentionPeriod": "3600",     # 1h; los eventos viejos no sirven
        "RedrivePolicy": json.dumps({"deadLetterTargetArn": dlq_arn, "maxReceiveCount": 3}),
    })["QueueUrl"]
    return url


if __name__ == "__main__":
    for q in QUEUES:
        print(f"{q}: {ensure_queue(q)}")
