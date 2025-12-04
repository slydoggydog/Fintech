import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  // Budget categories from the user's spreadsheet
  const categories = [
    // NEEDS
    { name: 'Groceries & Essentials', type: 'need', icon: '🛒', monthlyAmount: 1000 },
    { name: 'Property Tax', type: 'need', icon: '🏠', monthlyAmount: 2777.39 },
    { name: 'Auto Insurance', type: 'need', icon: '🚗', monthlyAmount: 205 },
    { name: 'Electricity & Gas', type: 'need', icon: '⚡', monthlyAmount: 300 },
    { name: 'Water', type: 'need', icon: '💧', monthlyAmount: 60 },
    { name: 'Sanitation/Garbage', type: 'need', icon: '🗑️', monthlyAmount: 30 },
    { name: 'Car Maintenance', type: 'need', icon: '🔧', monthlyAmount: 100 },
    { name: 'Gasoline', type: 'need', icon: '⛽', monthlyAmount: 300 },
    { name: 'Phone', type: 'need', icon: '📱', monthlyAmount: 40 },
    { name: 'Internet', type: 'need', icon: '🌐', monthlyAmount: 80 },
    { name: 'Loan Payments', type: 'need', icon: '💳', monthlyAmount: 480 },
    { name: 'Home Maintenance', type: 'need', icon: '🏡', monthlyAmount: 100 },
    { name: 'Daycare', type: 'need', icon: '👶', monthlyAmount: 980 },

    // WANTS
    { name: 'Clothing & Jewelry', type: 'want', icon: '👕', monthlyAmount: 50 },
    { name: 'Dining Out', type: 'want', icon: '🍽️', monthlyAmount: 200 },
    { name: 'Alcohol', type: 'want', icon: '🍺', monthlyAmount: 160 },
    { name: 'Entertainment', type: 'want', icon: '🎬', monthlyAmount: 20 },
    { name: 'Gym Membership', type: 'want', icon: '🏋️', monthlyAmount: 50 },
    { name: 'Streaming Services', type: 'want', icon: '📺', monthlyAmount: 80 },
    { name: 'Other Wants', type: 'want', icon: '🎁', monthlyAmount: 320 },

    // SAVINGS
    { name: 'Investments', type: 'savings', icon: '📈', monthlyAmount: 200 },
    { name: 'Emergency Fund', type: 'savings', icon: '🏦', monthlyAmount: 0 },

    // UNCATEGORIZED (for new transactions)
    { name: 'Uncategorized', type: 'need', icon: '❓', monthlyAmount: 0 },
  ];

  const currentMonth = new Date().getMonth() + 1;
  const currentYear = new Date().getFullYear();

  for (const cat of categories) {
    const category = await prisma.category.upsert({
      where: { name: cat.name },
      update: { type: cat.type, icon: cat.icon },
      create: { name: cat.name, type: cat.type, icon: cat.icon },
    });

    await prisma.budgetItem.upsert({
      where: {
        categoryId_month_year: {
          categoryId: category.id,
          month: currentMonth,
          year: currentYear,
        },
      },
      update: {
        monthlyAmount: cat.monthlyAmount,
        weeklyAmount: cat.monthlyAmount / 4.33,
      },
      create: {
        categoryId: category.id,
        monthlyAmount: cat.monthlyAmount,
        weeklyAmount: cat.monthlyAmount / 4.33,
        month: currentMonth,
        year: currentYear,
      },
    });
  }

  console.log('Seed completed!');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
