import { payload } from "./request.js";

/**
 * https://help.aliyun.com/zh/dashscope/developer-reference/api-details?spm=a2c4g.11186623.4.2.f11a3855r1pizQ&scm=20140722.H_2399481._.ID_2399481-OR_rec-V_1
 *
 */
export const GENERATION_URL =
  "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation";

export const chat = async (data) => {
  const { messages, history = [] } = data;
  const requestData = {
    model: "qwen-plus",
    input: {
      messages,
      history,
    },
    parameters: {},
  };
  return payload(GENERATION_URL, requestData);
};
