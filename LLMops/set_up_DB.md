# Setup Cosmos DB (MongoDB API) with Python

## 1️⃣ Get Your Connection Details

1. Go to the **Azure Portal** → Select your **Cosmos DB** resource.
2. On the **left menu**, go to **Connection String**.
3. Copy the following details:

   * **Primary connection string (MongoDB URI)**
   * **Username**
   * **Password**

Example connection string:

```
mongodb://<USERNAME>:<PASSWORD>@llmops-logdb.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb
```

> ⚠ Replace `<USERNAME>` and `<PASSWORD>` with the credentials provided in the portal.

---

## 2️⃣ Install MongoDB Python Driver

Install **PyMongo** to connect Python with Cosmos DB (MongoDB API):

```bash
pip install pymongo
```

---

## 3️⃣ Connect to Cosmos DB in Python