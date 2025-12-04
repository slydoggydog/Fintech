import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { syncTransactions } from '@/lib/plaid';

// Category mapping based on Plaid's category hierarchy
const CATEGORY_MAPPING: { [key: string]: string } = {
  'Food and Drink > Restaurants': 'Dining Out',
  'Food and Drink > Groceries': 'Groceries & Essentials',
  'Food and Drink > Coffee Shop': 'Dining Out',
  'Food and Drink > Bar': 'Alcohol',
  'Shops > Supermarkets and Groceries': 'Groceries & Essentials',
  'Shops > Clothing and Accessories': 'Clothing & Jewelry',
  'Travel > Gas': 'Gasoline',
  'Travel > Gas Stations': 'Gasoline',
  'Service > Insurance': 'Auto Insurance',
  'Service > Financial > Loans': 'Loan Payments',
  'Service > Utilities': 'Electricity & Gas',
  'Service > Internet Services': 'Internet',
  'Service > Phone': 'Phone',
  'Recreation > Gyms and Fitness Centers': 'Gym Membership',
  'Recreation > Arts and Entertainment': 'Entertainment',
  'Transfer > Debit': 'Uncategorized',
};

function mapCategory(plaidCategory: string[]): string {
  const categoryPath = plaidCategory.join(' > ');

  for (const [pattern, mappedCategory] of Object.entries(CATEGORY_MAPPING)) {
    if (categoryPath.startsWith(pattern)) {
      return mappedCategory;
    }
  }

  return 'Uncategorized';
}

export async function POST() {
  try {
    const plaidItems = await prisma.plaidItem.findMany();
    let totalAdded = 0;
    let totalModified = 0;

    for (const item of plaidItems) {
      let hasMore = true;
      let cursor = item.cursor || undefined;

      while (hasMore) {
        const { added, modified, removed, hasMore: more, nextCursor } =
          await syncTransactions(item.accessToken, cursor);

        // Process added transactions
        for (const transaction of added) {
          const categoryName = mapCategory(transaction.category || []);
          const category = await prisma.category.findUnique({
            where: { name: categoryName },
          });

          const account = await prisma.bankAccount.findUnique({
            where: { plaidAccountId: transaction.account_id },
          });

          if (account) {
            await prisma.transaction.upsert({
              where: { plaidTransactionId: transaction.transaction_id },
              update: {
                amount: transaction.amount,
                name: transaction.name,
                merchantName: transaction.merchant_name || null,
                date: new Date(transaction.date),
                pending: transaction.pending,
                categoryId: category?.id,
              },
              create: {
                plaidTransactionId: transaction.transaction_id,
                accountId: account.id,
                amount: transaction.amount,
                name: transaction.name,
                merchantName: transaction.merchant_name || null,
                date: new Date(transaction.date),
                pending: transaction.pending,
                categoryId: category?.id,
              },
            });
            totalAdded++;
          }
        }

        // Process modified transactions
        for (const transaction of modified) {
          await prisma.transaction.update({
            where: { plaidTransactionId: transaction.transaction_id },
            data: {
              amount: transaction.amount,
              name: transaction.name,
              merchantName: transaction.merchant_name || null,
              pending: transaction.pending,
            },
          });
          totalModified++;
        }

        // Process removed transactions
        for (const removal of removed) {
          await prisma.transaction.delete({
            where: { plaidTransactionId: removal.transaction_id },
          }).catch(() => {}); // Ignore if not found
        }

        // Update cursor
        await prisma.plaidItem.update({
          where: { id: item.id },
          data: { cursor: nextCursor },
        });

        cursor = nextCursor;
        hasMore = more;
      }
    }

    return NextResponse.json({
      success: true,
      added: totalAdded,
      modified: totalModified
    });
  } catch (error) {
    console.error('Sync error:', error);
    return NextResponse.json({ error: 'Sync failed' }, { status: 500 });
  }
}
