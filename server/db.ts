import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from '@shared/schema';

// Initialize the database connection using environment variables
const client = postgres(process.env.DATABASE_URL!);

// Create the database instance
export const db = drizzle(client, { schema });