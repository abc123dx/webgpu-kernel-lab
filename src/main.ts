import "./ui/styles.css";
import { KernelLabApp } from "./ui/app";

const root = document.querySelector<HTMLElement>("#app");
if (!root) {
  throw new Error("未找到应用挂载节点");
}

const app = new KernelLabApp(root);
void app.start();
