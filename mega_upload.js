const fs = require("fs");
const path = require("path");
// Debian's Node build does not always expose Web Crypto globally, while
// MEGAJS uses crypto.getRandomValues for client-side encryption.
if (!globalThis.crypto) globalThis.crypto = require("crypto").webcrypto;
const { Storage } = require("/opt/mega/node_modules/megajs");

async function upload() {
  const localPath = process.argv[2];
  if (!localPath || !fs.statSync(localPath).isFile()) {
    throw new Error("Expected a generated file to upload.");
  }

  const storage = await new Storage({
    email: process.env.MEGA_EMAIL,
    password: process.env.MEGA_PASSWORD,
    keepalive: false,
  }).ready;

  try {
    let folder = storage.root.children.find(
      (entry) => entry.directory && entry.name === "AI generations"
    );
    if (!folder) folder = await storage.root.mkdir("AI generations");

    const name = path.basename(localPath);
    await folder.upload({ name, size: fs.statSync(localPath).size }, fs.createReadStream(localPath)).complete;
    process.stdout.write(name);
  } finally {
    storage.close();
  }
}

upload().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
