import fetch from "node-fetch";
import { payload, query } from "./request.js";

async function sleep(time) {
  return new Promise((r) => setTimeout(r, time));
}

export async function checkTaskStatus(taskId) {
  const url = `https://dashscope.aliyuncs.com/api/v1/tasks/${taskId}`;
  const result = await query(url, { method: "GET" });

  if (response.ok) {
    const result = await response.json();
    const taskStatus = result.task_status;

    switch (taskStatus) {
      case "PENDING":
      case "RUNNING":
        // Task is still in progress, continue polling
        console.log("polling");
        await sleep(500);
        return checkTaskStatus(taskId);
        break;
      case "SUCCEEDED":
        console.log("Task succeeded. Results:", result.results);
        // Handle successful completion and retrieve image URL from results
        const results = result.results;
        return results;
        break;
      case "FAILED":
        console.log("Task failed. Reason:", result.results);
        // Handle failure and print failure reason
        return result;
        break;
      case "UNKNOWN":
        console.log(
          "Task does not exist or status is unknown. Reason:",
          result.results,
        );
        return result;
        // Handle unknown task or exception reason
        break;
      default:
        console.log("Unknown task status:", taskStatus);
        // Handle unknown task status
        break;
    }
  } else {
    console.log(`Request failed with status ${response.status}`);
  }
}

export async function createImage(data) {
  const url =
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis";

  const { prompt, style } = data;
  // Example usage
  const requestData = {
    model: "wanx-v1",
    input: {
      prompt: prompt || "",
    },
    parameters: {
      style: style || "<3d cartoon>",
      size: "1024*1024",
      n: 4,
      seed: 42,
    },
    ...data,
  };
  return payload(url, requestData);
}
