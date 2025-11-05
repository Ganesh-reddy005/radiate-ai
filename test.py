from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
  model="gpt-5-nano",
  input="write a haiku about ai",
  store=True,
)

print(response.output_text);
