db = db.getSiblingDB('mongodb');

db.tokens.insertOne({
    description: "Initial document"
});
