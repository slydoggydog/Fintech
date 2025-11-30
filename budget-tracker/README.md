# Family Budget Tracker

A personal finance tracking web application for married couples that connects to Bank of America accounts, tracks daily spending, and helps manage budgets.

## Features

- 💳 **Bank Account Integration**: Connects to Bank of America (checking and credit card) via Plaid API
- 📊 **Daily Spending Dashboard**: Review daily transactions and see how spending fits within budget
- 📈 **Real-time Budget Tracking**: Categorized spending with visual progress indicators
- ✏️ **Editable Budgets**: Adjust budget amounts on-the-fly
- 📅 **Historical Review**: Navigate through past days to review spending patterns
- 🎯 **50/30/20 Budget Framework**: Pre-configured with Needs, Wants, and Savings categories

## Tech Stack

- **Frontend**: React with TypeScript, Next.js 16
- **Backend**: Next.js API Routes
- **Database**: SQLite with Prisma ORM
- **Bank Connection**: Plaid API
- **Styling**: Tailwind CSS

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- A Plaid account (free sandbox available at https://dashboard.plaid.com)

### Installation

1. **Install dependencies** (already done if you're reading this):
   ```bash
   npm install
   ```

2. **Set up environment variables**:

   The `.env.local` file has been created with placeholders. Update it with your Plaid credentials:

   ```env
   # Get these from https://dashboard.plaid.com
   PLAID_CLIENT_ID=your_client_id_here
   PLAID_SECRET=your_sandbox_secret_here
   PLAID_ENV=sandbox  # Use 'development' for real Bank of America connection

   # Database (this is already correct)
   DATABASE_URL="file:./budget.db"

   # App Security (change this PIN)
   APP_PIN=1234
   ```

3. **Generate Prisma Client and create database**:
   ```bash
   npx prisma generate
   npx prisma db push
   ```

4. **Seed the database with your budget categories**:
   ```bash
   npm run db:seed
   ```

5. **Start the development server**:
   ```bash
   npm run dev
   ```

6. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## Budget Categories

Your budget has been pre-configured with the following categories:

### 🏠 Needs ($6,452.39/month - 71.7% of income)
- Groceries & Essentials: $1,000
- Property Tax: $2,777.39
- Auto Insurance: $205
- Electricity & Gas: $300
- Water: $60
- Sanitation/Garbage: $30
- Car Maintenance: $100
- Gasoline: $300
- Phone: $40
- Internet: $80
- Loan Payments: $480
- Home Maintenance: $100
- Daycare: $980

### 🎁 Wants ($880/month - 9.8% of income)
- Clothing & Jewelry: $50
- Dining Out: $200
- Alcohol: $160
- Entertainment: $20
- Gym Membership: $50
- Streaming Services: $80
- Other Wants: $320

### 💰 Savings ($200/month - 2.2% of income)
- Investments: $200
- Emergency Fund: $0

**Monthly Income**: $9,000
**Remaining After Budget**: $1,467.61

## Connecting Your Bank Accounts

### Option 1: Sandbox Testing (Recommended First)

1. Start the app and navigate to the accounts page
2. Click "Connect Bank Account"
3. In the Plaid Link interface, use these test credentials:
   - Institution: Search for "Bank of America"
   - Username: `user_good`
   - Password: `pass_good`
4. Select accounts to link
5. Test transactions will appear automatically

### Option 2: Development Mode (Real Bank Connection)

1. Sign up for Plaid Development tier at https://dashboard.plaid.com
   - Free for up to 100 connected items
   - Requires company verification (1-2 days)

2. Update `.env.local`:
   ```env
   PLAID_ENV=development
   PLAID_CLIENT_ID=your_dev_client_id
   PLAID_SECRET=your_dev_secret
   ```

3. Connect your real Bank of America accounts through the app

### Option 3: Production (Full Access)

For production use with unlimited accounts:
- Costs $500/month + per-user fees
- Contact Plaid sales for setup

## Usage Guide

### Daily Review Workflow

1. **Open the app each morning or evening**
2. **Review yesterday's transactions** on the Daily View
3. **Check if you stayed within budget** (green = good, red = over)
4. **Discuss spending together** as a couple
5. **Adjust categories** if transactions were auto-categorized incorrectly

### Syncing Transactions

Transactions sync automatically when you visit the app, but you can manually trigger a sync:

```bash
# Call the sync API endpoint
curl -X POST http://localhost:3000/api/transactions/sync
```

Or set up a cron job to sync daily:
```bash
0 6 * * * curl -X POST http://localhost:3000/api/transactions/sync
```

### Adjusting Your Budget

1. Navigate to **Budget Settings**
2. Click on any budget amount to edit it
3. Enter the new monthly amount
4. Click **Save**
5. The weekly amount updates automatically (monthly ÷ 4.33)

## Project Structure

```
budget-tracker/
├── app/
│   ├── api/                    # API Routes
│   │   ├── plaid/              # Plaid integration endpoints
│   │   ├── budget/             # Budget CRUD operations
│   │   ├── spending/           # Spending analytics
│   │   └── transactions/       # Transaction sync
│   ├── components/             # React Components
│   │   ├── DailyDashboard.tsx  # Daily spending view
│   │   └── BudgetManager.tsx   # Budget configuration
│   ├── budget/                 # Budget settings page
│   ├── layout.tsx              # Root layout with navigation
│   └── page.tsx                # Home page (daily view)
├── lib/
│   ├── db.ts                   # Prisma client singleton
│   └── plaid.ts                # Plaid API wrapper
├── prisma/
│   ├── schema.prisma           # Database schema
│   └── seed.ts                 # Budget category seed data
└── .env.local                  # Environment variables
```

## Database Schema

The app uses 5 main tables:

1. **BankAccount**: Linked bank accounts from Plaid
2. **Transaction**: All spending transactions
3. **Category**: Budget categories (Groceries, Gas, etc.)
4. **BudgetItem**: Monthly budget amounts per category
5. **PlaidItem**: Plaid connection metadata

## API Endpoints

### Plaid Integration
- `POST /api/plaid/create-link-token` - Generate Plaid Link token
- `POST /api/plaid/exchange-token` - Exchange public token for access token

### Budget Management
- `GET /api/budget` - Get all budget items with spending data
- `PUT /api/budget` - Update budget amount for a category

### Spending Analytics
- `GET /api/spending/daily?date=YYYY-MM-DD` - Get daily spending summary

### Transaction Sync
- `POST /api/transactions/sync` - Manually sync transactions from Plaid

## Customization

### Adding a New Budget Category

1. Add the category to the database:
   ```sql
   INSERT INTO Category (id, name, type, icon)
   VALUES ('new_id', 'Category Name', 'need', '🏷️');
   ```

2. Create a budget item for the current month:
   ```sql
   INSERT INTO BudgetItem (categoryId, monthlyAmount, weeklyAmount, month, year)
   VALUES ('new_id', 100, 23.09, 12, 2025);
   ```

3. Or modify `prisma/seed.ts` and re-run the seed

### Changing the Monthly Income

Edit `/app/components/BudgetManager.tsx`:
```typescript
const monthlyIncome = 9000; // Change this value
```

### Customizing Transaction Categories

Edit the `CATEGORY_MAPPING` object in `/app/api/transactions/sync/route.ts` to change how Plaid categories map to your budget categories.

## Deployment Options

### Option 1: Vercel (Recommended)

1. Push your code to GitHub
2. Import to Vercel
3. Add environment variables in Vercel dashboard
4. Note: You'll need to upgrade the database to PostgreSQL for production

### Option 2: Railway

1. Connect your GitHub repo
2. Railway auto-detects Next.js
3. Add environment variables
4. Deploy

### Option 3: Self-Hosted (Raspberry Pi)

Perfect for keeping financial data private:

1. Install Node.js on your Pi
2. Clone the repository
3. Run `npm install && npm run build`
4. Use PM2 to keep it running: `pm2 start npm --name budget -- start`
5. Access on your local network

## Security Considerations

⚠️ **Important Security Notes**:

1. **Never commit `.env.local`** to version control
2. **Change the APP_PIN** from the default
3. **Use HTTPS in production** (Vercel provides this automatically)
4. **Plaid access tokens are sensitive** - they're stored encrypted in the database
5. **Backup your database regularly** - it contains all your financial data
6. **Consider adding authentication** beyond the simple PIN for internet-exposed deployments

## Troubleshooting

### "Failed to fetch Prisma engines"
If you see this error during `npx prisma generate`:
```bash
# Try with checksum bypass
PRISMA_ENGINES_CHECKSUM_IGNORE_MISSING=1 npx prisma generate
```

### Transactions not appearing
1. Check that your bank accounts are connected in the database
2. Manually trigger a sync: `curl -X POST http://localhost:3000/api/transactions/sync`
3. Check Plaid dashboard for any connection issues

### Budget not loading
Ensure you've run the seed command:
```bash
npm run db:seed
```

### Plaid Link not opening
Check that your Plaid credentials in `.env.local` are correct and that `PLAID_ENV` matches your account type.

## Development Roadmap

Future enhancements you might want to add:

- [ ] Weekly and monthly summary views
- [ ] Charts and graphs using Recharts
- [ ] Email/SMS alerts when approaching budget limits
- [ ] Multi-currency support
- [ ] Receipt photo uploads
- [ ] Recurring transaction predictions
- [ ] Custom category creation from UI
- [ ] Export to CSV/Excel
- [ ] Mobile app (React Native)

## Support

For Plaid-related issues:
- Plaid Documentation: https://plaid.com/docs/
- Plaid Support: https://support.plaid.com/

For Next.js issues:
- Next.js Documentation: https://nextjs.org/docs

## License

This is a personal project. Feel free to fork and modify for your own use.

## Acknowledgments

Built with:
- Next.js 16
- Prisma ORM
- Plaid API
- Tailwind CSS
- date-fns
- Recharts (for future graphs)
