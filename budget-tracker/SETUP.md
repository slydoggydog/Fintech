# Setup Instructions

Complete these steps to get your Family Budget Tracker running.

## Prerequisites

✅ Node.js 18+ installed
✅ npm installed
✅ Budget tracker code downloaded (you have this!)

## Step-by-Step Setup

### 1. Get Plaid API Credentials

1. Go to https://dashboard.plaid.com
2. Sign up for a free account
3. Create a new application
4. Copy your credentials:
   - Client ID
   - Sandbox Secret (for testing)
   - Or Development Secret (for real Bank of America connection)

### 2. Configure Environment Variables

Edit the `.env.local` file in the project root:

```env
PLAID_CLIENT_ID=paste_your_client_id_here
PLAID_SECRET=paste_your_secret_here
PLAID_ENV=sandbox  # or 'development' for real banks
DATABASE_URL="file:./budget.db"
APP_PIN=1234  # Change this to your preferred PIN
```

### 3. Generate Database and Prisma Client

Run these commands in the budget-tracker directory:

```bash
# Generate the Prisma client
npx prisma generate

# Create the SQLite database with all tables
npx prisma db push

# Seed the database with your budget categories
npm run db:seed
```

You should see output like:
```
✓ Generated Prisma Client
Database synchronized
Seed completed!
```

### 4. Start the Development Server

```bash
npm run dev
```

The app will be available at http://localhost:3000

### 5. Connect Your Bank Account (Optional - Testing)

1. Open http://localhost:3000
2. Click on "Bank Accounts" in the navigation
3. Click "Connect Bank Account"
4. In the Plaid Link popup:
   - Search for "Bank of America"
   - Username: `user_good`
   - Password: `pass_good`
   - Select accounts
5. Test transactions will appear automatically

## Troubleshooting

### "Module not found: Can't resolve '.prisma/client'"

You need to run `npx prisma generate` first.

### "Error: P1003: Database does not exist"

Run `npx prisma db push` to create the database.

### "Seed failed" or "No categories found"

Make sure the database was created first with `npx prisma db push`, then run `npm run db:seed`.

### Network Errors with Prisma

If you see "Failed to fetch Prisma engines":

```bash
# Set this environment variable
export PRISMA_ENGINES_CHECKSUM_IGNORE_MISSING=1
npx prisma generate
```

Or on Windows:
```cmd
set PRISMA_ENGINES_CHECKSUM_IGNORE_MISSING=1
npx prisma generate
```

### Plaid Link Not Opening

1. Check that PLAID_CLIENT_ID and PLAID_SECRET are set correctly in `.env.local`
2. Make sure PLAID_ENV matches your credentials type (sandbox, development, or production)
3. Restart the dev server after changing .env.local

## Verifying Everything Works

After setup, you should be able to:

1. ✅ Visit http://localhost:3000 and see the Daily Dashboard
2. ✅ Navigate to Budget Settings and see all your categories
3. ✅ Edit a budget amount and save it
4. ✅ Navigate between days on the Daily Dashboard
5. ✅ (With Plaid configured) Connect a bank account

## Next Steps

1. **Test with Sandbox**: Connect a test bank account using Plaid sandbox
2. **Review Budget**: Check if the pre-configured budget categories match your needs
3. **Adjust Categories**: Modify amounts in Budget Settings
4. **Go Live**: When ready, switch to Plaid Development mode for real bank connections

## Need Help?

- Check the main README.md for detailed documentation
- Review Plaid documentation: https://plaid.com/docs/
- Check Next.js docs: https://nextjs.org/docs

## Quick Reference

| Command | Purpose |
|---------|---------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm start` | Run production build |
| `npx prisma studio` | Open database GUI |
| `npx prisma db push` | Update database schema |
| `npm run db:seed` | Reseed budget categories |

## Production Deployment

When ready to deploy:

1. Push code to GitHub
2. Deploy to Vercel (recommended) or Railway
3. Add environment variables in hosting dashboard
4. For Vercel, upgrade to PostgreSQL database
5. Run migrations and seed in production

See README.md for detailed deployment instructions.
