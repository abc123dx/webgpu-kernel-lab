import "./ui/styles.css";
import { KernelLabApp } from "./ui/app";

const root = document.querySelector<HTMLElement>("#app");
if (!root) {
  throw new Error("Application root was not found");
}

const app = new KernelLabApp(root);
void app.start();
