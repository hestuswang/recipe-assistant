import fetch from "node-fetch";

let API_KEY = process.env.DASHSCOPE_API_KEY;

export const config = (apiKey = API_KEY) => {
  API_KEY = key;
};

const query = async (url, data, apiKey = API_KEY) => {
  const headers = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };
  return fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify(data),
  }).then((response) => response.json());
};

const payload = async (url, data, apiKey = API_KEY) => {
  const headers = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };
  return fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify(data),
  }).then((response) => response.json());
};

export { query, payload };
