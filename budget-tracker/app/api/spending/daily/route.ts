import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { startOfDay, endOfDay, parseISO, getDaysInMonth } from 'date-fns';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const dateStr = searchParams.get('date') || new Date().toISOString().split('T')[0];
  const date = parseISO(dateStr);

  const dayStart = startOfDay(date);
  const dayEnd = endOfDay(date);

  // Get transactions for the day
  const transactions = await prisma.transaction.findMany({
    where: {
      date: {
        gte: dayStart,
        lte: dayEnd,
      },
      amount: { gt: 0 }, // Only spending (positive amounts)
    },
    include: {
      category: true,
    },
    orderBy: { date: 'desc' },
  });

  // Calculate totals
  const total = transactions.reduce((sum, t) => sum + t.amount, 0);

  // Group by category
  const byCategory: { [key: string]: number } = {};
  transactions.forEach((t) => {
    const catName = t.category?.name || 'Uncategorized';
    byCategory[catName] = (byCategory[catName] || 0) + t.amount;
  });

  // Get budget for comparison
  const month = date.getMonth() + 1;
  const year = date.getFullYear();
  const daysInMonth = getDaysInMonth(date);

  const budgetItems = await prisma.budgetItem.findMany({
    where: { month, year },
    include: { category: true },
  });

  const monthlyBudget = budgetItems.reduce((sum, item) => sum + item.monthlyAmount, 0);
  const dailyTarget = monthlyBudget / daysInMonth;

  return NextResponse.json({
    daily: {
      date: dateStr,
      transactions,
      total,
      byCategory,
    },
    budgetComparison: {
      dailyTarget,
      monthlyBudget,
      daysInMonth,
    },
  });
}
