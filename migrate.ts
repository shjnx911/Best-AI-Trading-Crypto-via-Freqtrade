import { drizzle } from 'drizzle-orm/postgres-js';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import postgres from 'postgres';

async function main() {
  const client = postgres(process.env.DATABASE_URL!);

  const db = drizzle(client);

  console.log('Running migrations...');
  
  await migrate(db, { migrationsFolder: './migrations' });
  
  console.log('Migrations complete!');
  
  // Close the connection
  await client.end();
}

main().catch((err) => {
  console.error('Error during migration:', err);
  process.exit(1);
});